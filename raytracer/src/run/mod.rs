// src/run/mod.rs

use std::path::Path;
use std::io::Result as IoResult;

use chrono::Local;
use serde::Deserialize;

use crate::prelude::*;
use crate::camera::Camera;
use crate::image::output::write_ppm;
#[cfg(target_arch = "aarch64")]
use crate::image::output::write_ppm_neon;

use crate::scene::{SceneKind, build_scene};
use crate::metrics::runlog::write_metrics_to_file;
use crate::metrics::{RunMetrics, MetricsCollector};

use crate::render::{RenderParams, BackendKind, make_renderer_for_scene};

use crate::world::sphere::Sphere;
use crate::world::material::MaterialKind;

/// Configuración de una corrida de render.
/// `scene_seed` permite fijar la aleatoriedad de la escena (por ejemplo ManySpheres).
#[derive(Debug, Deserialize)]
pub struct RunConfig {
    pub backend: BackendKind,
    pub scene: SceneKind,
    pub image_width: i32,
    pub aspect_ratio: f64,
    pub samples_per_pixel: i32,
    pub max_depth: i32,
    /// Semilla opcional para la construcción de la escena.
    /// - `Some(seed)` -> escena determinista.
    /// - `None`       -> escena con aleatoriedad "libre".
    pub scene_seed: Option<u64>,
}

/// Helper interno para construir un identificador único de la corrida.
/// - fecha y hora local (YYYYMMDD_HHMMSS)
/// - backend (scalar / neon)
/// - resolución (width x height)
/// - samples per pixel
fn build_run_id(
    backend: BackendKind,
    image_width: i32,
    image_height: i32,
    samples_per_pixel: i32,
) -> String {
    let now = Local::now();
    let timestamp = now.format("%Y%m%d_%H%M%S").to_string();

    let backend_tag = match backend {
        BackendKind::Scalar => "scalar",
        BackendKind::Neon   => "neon",
    };

    format!(
        "{}_{}_{}x{}_spp{}",
        timestamp,
        backend_tag,
        image_width,
        image_height,
        samples_per_pixel
    )
}

/// Cuenta cuántas esferas hay en total en la escena y cuántas hay
/// de cada tipo de material conocido.
///
/// Parameters:
/// - `spheres`: slice de esferas planas usadas para aceleración.
///
/// Returns:
/// - tupla (total, lambertian, metal, dielectric)
fn compute_sphere_material_counts(spheres: &[Sphere]) -> (u64, u64, u64, u64) {
    let mut total: u64 = 0;
    let mut lambertian: u64 = 0;
    let mut metal: u64 = 0;
    let mut dielectric: u64 = 0;

    for s in spheres {
        total += 1;

        match s.material_kind() {
            MaterialKind::Lambertian => lambertian += 1,
            MaterialKind::Metal      => metal += 1,
            MaterialKind::Dielectric => dielectric += 1,
            MaterialKind::Other      => {
                // Se dejan otros materiales fuera del desglose específico.
            }
        }
    }

    (total, lambertian, metal, dielectric)
}


/// Ejecuta una corrida completa:
/// - Construye la escena según el tipo (World + Vec<Sphere>).
/// - Configura la cámara.
/// - Prepara RenderParams y framebuffer.
/// - Elige backend y construye renderer (con o sin aceleración de esferas).
/// - Mide el render con MetricsCollector.
/// - Escribe imagen (PPM) y archivo de métricas, ambos con el mismo run_id.
/// - Retorna las métricas de la corrida.
pub fn execute_run(config: &RunConfig) -> IoResult<RunMetrics> {
    // ---------- Escena (World + Vec<Sphere>) ----------
    //
    // build_scene devuelve:
    //   - world: HittableList usado por el camino escalar.
    //   - spheres: Vec<Sphere> plano usado por NEON para world_hit4_spheres.
    // `scene_seed` permite fijar la aleatoriedad (por ejemplo ManySpheres).
    let scene_data = build_scene(config.scene, config.scene_seed);
    let world = scene_data.world;
    let spheres_for_accel = scene_data.spheres;

    // Se cuentan las esferas por tipo de material para las métricas.
    let (
        scene_spheres_total,
        scene_spheres_lambertian,
        scene_spheres_metal,
        scene_spheres_dielectric,
    ) = compute_sphere_material_counts(&spheres_for_accel);

    // ---------- Camera ----------
    let mut cam = Camera::new(config.image_width, config.aspect_ratio);
    cam.samples_per_pixel = config.samples_per_pixel;
    cam.max_depth = config.max_depth;

    // Parámetros de cámara (por ahora fijos, más adelante pueden ir también en RunConfig)
    cam.vfov = 20.0;
    cam.lookfrom = Point3::new(13.0, 2.0, 3.0);
    cam.lookat   = Point3::new( 0.0, 0.0, 0.0);
    cam.vup      = Vec3::new(0.0, 1.0, 0.0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    // ---------- Render Params ----------
    let params = RenderParams {
        image_width: cam.image_width,
        aspect_ratio: cam.aspect_ratio,
        samples_per_pixel: cam.samples_per_pixel,
        max_depth: cam.max_depth,
    };

    // Altura de imagen estimada (igual que en Camera::initialize).
    let mut image_height = (params.image_width as f64 / params.aspect_ratio) as i32;
    if image_height < 1 {
        image_height = 1;
    }

    // Framebuffer en memoria: un Color por píxel.
    let mut framebuffer = vec![
        Color::new(0.0, 0.0, 0.0);
        (params.image_width * image_height) as usize
    ];

    // ---------- Render backend ----------
    let backend_kind = config.backend;

    // Para el backend escalar no necesitamos aceleración.
    // Para el backend NEON, pasamos el Vec<Sphere> que construimos en build_scene.
    let sphere_accel = match backend_kind {
        BackendKind::Scalar => None,
        BackendKind::Neon   => Some(spheres_for_accel),
    };

    // make_renderer recibe ahora:
    //   - qué backend usar (Scalar / Neon),
    //   - y opcionalmente el Vec<Sphere> para aceleración de intersecciones.
    let mut renderer = make_renderer_for_scene(config.backend, sphere_accel);

    // Se construye el run_id usando la resolución planeada y spp.
    let run_id = build_run_id(
        backend_kind,
        params.image_width,
        image_height,
        params.samples_per_pixel,
    );

    // Se mide el render con MetricsCollector.
    let mut metrics: RunMetrics = MetricsCollector::measure_render(
        renderer.as_mut(), // &mut dyn Renderer
        backend_kind,
        run_id.clone(),
        &params,
        &world,
        &mut cam,
        &mut framebuffer,
    );


    // Se asocia la semilla de escena a las métricas de esta corrida.
    metrics.scene_seed = config.scene_seed;

    // Se rellenan las estadísticas de escena con los conteos calculados.
    metrics.scene_spheres_total = scene_spheres_total;
    metrics.scene_spheres_lambertian = scene_spheres_lambertian;
    metrics.scene_spheres_metal = scene_spheres_metal;
    metrics.scene_spheres_dielectric = scene_spheres_dielectric;
    
    // Altura efectiva de imagen (por si cambiara en initialize).
    let final_image_height = metrics.image_height;

    // ---------- Output de imagen ----------
    let image_filename = format!("output/{}.ppm", metrics.run_id);
    let image_path = Path::new(&image_filename);

    match backend_kind {
        BackendKind::Scalar => {
            // Camino 100% escalar: renderer escalar + postprocesado escalar.
            write_ppm(
                image_path,
                params.image_width,
                final_image_height,
                &framebuffer,
            )?;
        }

        BackendKind::Neon => {
            // Si está en aarch64, se usa el writer NEON (postprocesado SIMD).
            // En otras arquitecturas, se hace el fallback al writer escalar.
            #[cfg(target_arch = "aarch64")]
            {
                write_ppm_neon(
                    image_path,
                    params.image_width,
                    final_image_height,
                    &framebuffer,
                )?;
            }

            #[cfg(not(target_arch = "aarch64"))]
            {
                // Fallback seguro: mismo comportamiento que el camino escalar.
                write_ppm(
                    image_path,
                    params.image_width,
                    final_image_height,
                    &framebuffer,
                )?;
            }
        }
    }

    // ---------- Output de métricas ----------
    let metrics_filename = format!("runs/{}.txt", metrics.run_id);
    let metrics_path = Path::new(&metrics_filename);
    write_metrics_to_file(&metrics_path, &metrics)?;

    // ---------- Log en consola ----------
    eprintln!(
        "\nRun {} completed: \
         \nbackend={:?}, \
         \nresolution={}x{}, \
         \nspp={}, \
         \nmax_depth={}, \
         \nrender_time_ms={}",
        metrics.run_id,
        metrics.backend,
        metrics.image_width,
        metrics.image_height,
        metrics.samples_per_pixel,
        metrics.max_depth,
        metrics.render_duration_ms,
    );

    Ok(metrics)
}
