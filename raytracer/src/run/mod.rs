// src/run/mod.rs

use std::path::Path;
use std::io::Result as IoResult;

use chrono::Local;

use crate::prelude::*; // Color, Point3, Vec3
use crate::camera::Camera;
use crate::render::{RenderParams, BackendKind, make_renderer};
use crate::output::write_ppm;
use crate::metrics::{RunMetrics, MetricsCollector};
use crate::metrics::runlog::write_metrics_to_file;
use crate::scene::{SceneKind, build_world};

/// Configuración de una corrida de render.
/// Más adelante aquí se puede agregar scene_seed.
pub struct RunConfig {
    pub backend: BackendKind,
    pub scene: SceneKind,
    pub image_width: i32,
    pub aspect_ratio: f64,
    pub samples_per_pixel: i32,
    pub max_depth: i32,
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

/// Ejecuta una corrida completa:
/// - Construye el mundo según la escena.
/// - Configura la cámara.
/// - Prepara RenderParams y framebuffer.
/// - Elige backend y construye renderer.
/// - Mide el render con MetricsCollector.
/// - Escribe imagen (PPM) y archivo de métricas, ambos con el mismo run_id.
/// - Retorna las métricas de la corrida.
pub fn execute_run(config: &RunConfig) -> IoResult<RunMetrics> {
    // ---------- World ----------
    let world = build_world(config.scene);

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
    if image_height < 1 { image_height = 1; }

    // Framebuffer en memoria: un Color por píxel.
    let mut framebuffer = vec![
        Color::new(0.0, 0.0, 0.0);
        (params.image_width * image_height) as usize
    ];

    // ---------- Render backend ----------
    let backend_kind = config.backend;
    let mut renderer = make_renderer(backend_kind);

    // Se contruye el run_id usando la resolución planeada y spp.
    let run_id = build_run_id(
        backend_kind,
        params.image_width,
        image_height,
        params.samples_per_pixel,
    );

    // Se mide el render con MetricsCollector.
    let metrics: RunMetrics = MetricsCollector::measure_render(
        renderer.as_mut(), // &mut dyn Renderer
        backend_kind,
        run_id.clone(),
        &params,
        &world,
        &mut cam,
        &mut framebuffer,
    );

    // Altura efectiva de imagen (por si cambiara en initialize).
    let final_image_height = metrics.image_height;

    // ---------- Output de imagen ----------
    let image_filename = format!("output/{}.ppm", metrics.run_id);
    let image_path = Path::new(&image_filename);
    write_ppm(image_path, params.image_width, final_image_height, &framebuffer)?;

    // ---------- Output de métricas ----------
    let metrics_filename = format!("runs/{}.txt", metrics.run_id);
    let metrics_path = Path::new(&metrics_filename);
    write_metrics_to_file(&metrics_path, &metrics)?;

    // ---------- Log en consola ----------
    eprintln!(
        "\nRun {} completed: \nbackend={:?}, \nresolution={}x{}, \nspp={}, \nmax_depth={}, \nrender_time_ms={}",
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
