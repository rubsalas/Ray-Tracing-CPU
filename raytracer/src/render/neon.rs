// src/render/neon.rs

//! NEON backend for the ray tracer.
//!
//! En este estado del pipeline, el backend NEON:
//! - Recorre la imagen en bloques de 4 píxeles en X.
//! - Genera rayos y calcula el color usando el camino escalar
//!   (camera.get_ray + camera.ray_color).
//! - Usa SIMD sólo para la parte de intersecciones rayo–esfera mediante
//!   world_hit4_spheres, pero SOLO como verificación en debug.
//!
//! Es decir, la imagen resultante es idéntica (a igualdad de randoms)
//! al backend escalar; simplemente estamos ejercitando y validando la
//! geometría vectorizada sin alterar aún el sombreado.

use crate::prelude::*;
use crate::camera::Camera;
use crate::world::hittable::{Hittable, HitRecord};
use crate::world::sphere::Sphere;
use crate::interval::Interval;

use crate::simd::neon::{Ray4, build_primary_rays_block};
use crate::simd::hit::{HitInfo4, world_hit4_spheres};

use super::{RenderParams, Renderer};
use super::ScalarRenderer;

use crate::render::primary::shade_primary_with_sphere_hint;

use std::any::Any;

/// Contadores básicos del backend NEON para entender cuánto se usa la aceleración.
#[derive(Debug, Default, Clone, Copy)]
pub struct NeonStats {
    /// Número total de rayos primarios procesados por este backend.
    pub primary_rays_total: u64,
    /// De los anteriores, cuántos usaron el camino acelerado con world_hit4_spheres.
    pub primary_rays_accelerated: u64,
    /// De los anteriores, cuántos terminaron usando el camino escalar de fallback.
    pub primary_rays_fallback: u64,
}

/// Renderer basado en NEON.
///
/// Por ahora:
/// - Usa bloques de 4 píxeles.
/// - Puede usar aceleración de intersección con `sphere_accel` en el primer rebote.
/// - Recurre al camino escalar (`scalar_fallback`) para todo lo demás.
pub struct NeonRenderer {
    /// Camino escalar de referencia (para fallback y partes no vectorizadas).
    scalar_fallback: ScalarRenderer,
    /// Copia plana de esferas para world_hit4_spheres (si se desea aceleración).
    sphere_accel: Option<Vec<Sphere>>,
    /// Contadores de uso de NEON vs fallback.
    pub stats: NeonStats,
}

impl NeonRenderer {
    /// Crea un nuevo backend NEON.
    ///
    /// - `sphere_accel`: si es `Some(Vec<Sphere>)`, se intentará usar world_hit4_spheres
    ///   para acelerar el primer rebote de los rayos primarios.
    /// - Si es `None`, se comporta casi igual que el backend escalar, pero iterando en
    ///   bloques de 4 y usando SIMD solo para postprocesado (según las tareas que ya hiciste).
    pub fn new(sphere_accel: Option<Vec<Sphere>>) -> Self {
        NeonRenderer {
            scalar_fallback: ScalarRenderer::new(),
            sphere_accel,
            stats: NeonStats::default(),
        }
    }

    /// En debug, compara el resultado de `world_hit4_spheres` contra un
    /// recorrido escalar clásico que llama `Sphere::hit` sobre todas las
    /// esferas para cada rayo en `ray4`.
    #[cfg(debug_assertions)]
    fn debug_check_world_hit4_vs_scalar(
        &self,
        ray4: &Ray4,
        spheres: &[Sphere],
        info: &HitInfo4,
    ) {
        // Recuperamos los 4 rayos escalares desde el pack SIMD.
        let rays = ray4.to_rays();

        for lane in 0..4 {
            let r = &rays[lane];

            let mut best_t = f64::INFINITY;
            let mut best_idx: i32 = -1;

            let t_min = 0.001_f64;
            let t_max = f64::INFINITY;
            let interval = Interval::new(t_min, t_max);
            let mut tmp_rec = HitRecord::default();

            // Camino escalar de referencia.
            for (idx, sphere) in spheres.iter().enumerate() {
                if sphere.hit(r, &interval, &mut tmp_rec) {
                    if tmp_rec.t < best_t {
                        best_t = tmp_rec.t;
                        best_idx = idx as i32;
                    }
                }
            }

            let simd_has_hit = info.hit[lane];
            let simd_idx = info.sphere_index[lane];
            let simd_t = info.t[lane];

            if best_idx == -1 {
                // Escalar: no hubo hit
                assert!(
                    !simd_has_hit,
                    "lane {}: scalar has NO hit, but NEON reports a hit",
                    lane
                );
                assert_eq!(
                    -1,
                    simd_idx,
                    "lane {}: scalar has NO hit, but NEON sphere_index != -1 (got {})",
                    lane,
                    simd_idx
                );
            } else {
                // Escalar: sí hubo hit
                assert!(
                    simd_has_hit,
                    "lane {}: scalar HAS hit, but NEON reports no hit",
                    lane
                );
                assert_eq!(
                    best_idx,
                    simd_idx,
                    "lane {}: sphere index mismatch scalar={} NEON={}",
                    lane,
                    best_idx,
                    simd_idx
                );

                let t_scalar = best_t as f32;
                let diff = (t_scalar - simd_t).abs();
                let eps = 1e-3_f32;
                assert!(
                    diff <= eps,
                    "lane {}: t mismatch scalar={} NEON={} diff={}",
                    lane,
                    t_scalar,
                    simd_t,
                    diff
                );
            }
        }
    }
}

impl Renderer for NeonRenderer {
    fn render(
        &mut self,
        world: &dyn Hittable,
        camera: &mut Camera,
        params: &RenderParams,
        framebuffer: &mut [Color],
    ) {
        // Se reinician los contadores para esta corrida.
        self.stats = NeonStats::default();

        // Se prepara la cámara igual que en el backend escalar.
        camera.initialize();

        let image_width       = params.image_width;
        let image_height      = camera.image_height();
        let samples_per_pixel = params.samples_per_pixel;
        let max_depth         = params.max_depth;
        let pixel_scale       = camera.pixel_samples_scale();

        assert_eq!(
            framebuffer.len(),
            (image_width * image_height) as usize,
            "framebuffer size mismatch in NeonRenderer"
        );

        // Si no se tiene aceleración de esferas configurada,
        // se delega todo al backend escalar para no alterar el comportamiento.
        let spheres = match &self.sphere_accel {
            Some(v) => v,
            None => {
                self.scalar_fallback
                    .render(world, camera, params, framebuffer);
                return;
            }
        };

        // Se recorre la imagen en bloques de 4 píxeles horizontales.
        for j in 0..image_height {
            eprint!("\rScanlines remaining (NEON): {} ", image_height - j);
            use std::io::Write as _;
            std::io::stderr().flush().ok();

            for i_block in (0..image_width).step_by(4) {
                // Se inicializa el acumulador de color lineal por lane (para las muestras).
                let mut lane_accum = [
                    Color::new(0.0, 0.0, 0.0),
                    Color::new(0.0, 0.0, 0.0),
                    Color::new(0.0, 0.0, 0.0),
                    Color::new(0.0, 0.0, 0.0),
                ];

                // Se realiza el multi-sampling por píxel.
                for _s in 0..samples_per_pixel {
                    // 1) Se construyen los 4 rayos primarios del bloque mediante el helper.
                    let (rays, ray4, lane_valid) =
                        build_primary_rays_block(camera, j, i_block, image_width);

                    // 2) Se cuenta cuántos lanes están activos en esta muestra.
                    let mut active_lanes = 0_u64;
                    for lane in 0..4 {
                        if lane_valid[lane] {
                            active_lanes += 1;
                        }
                    }
                    // Rayos primarios totales (por muestra).
                    self.stats.primary_rays_total += active_lanes;

                    // 3) Se calculan intersecciones primarias vectorizadas
                    //    usando solo esferas (world_hit4_spheres).
                    let t_min = 0.001_f32;
                    let t_max = f32::INFINITY;
                    let info: HitInfo4 = world_hit4_spheres(&ray4, spheres, t_min, t_max);

                    // 4) En modo debug, se verifica la coherencia contra un recorrido escalar.
                    #[cfg(debug_assertions)]
                    self.debug_check_world_hit4_vs_scalar(&ray4, spheres, &info);

                    // 5) Se sombrea escalar por lane, usando el hint SIMD
                    //    para el primer rebote a través de shade_primary_with_sphere_hint.
                    let mut accelerated_lanes = 0_u64;

                    for lane in 0..4 {
                        if !lane_valid[lane] {
                            continue;
                        }

                        let has_primary_hit = info.hit[lane];
                        if has_primary_hit {
                            accelerated_lanes += 1;
                        }

                        let sample_color = shade_primary_with_sphere_hint(
                            camera,
                            world,
                            &rays[lane],
                            max_depth,
                            has_primary_hit,
                            info.sphere_index[lane],
                            spheres,
                        );

                        lane_accum[lane] += sample_color;
                    }

                    // Se actualizan contadores por muestra.
                    self.stats.primary_rays_accelerated += accelerated_lanes;
                    self.stats.primary_rays_fallback += active_lanes - accelerated_lanes;
                }

                // 6) Al final del muestreo, se aplica 1/spp y
                //    se escribe al framebuffer en forma lineal.
                for lane in 0..4 {
                    let ix = i_block + lane as i32;
                    if ix >= image_width {
                        continue;
                    }
                    let idx = (j * image_width + ix) as usize;

                    // Igual que en ScalarRenderer: se deja el valor lineal ya
                    // multiplicado por pixel_scale (= 1/samples_per_pixel).
                    framebuffer[idx] = pixel_scale * lane_accum[lane];
                }
            }
        }

        eprintln!(
            "\rDone (NEON primary-hit acceleration). \
             \n[NEON stats] primary_rays_total={}, accelerated={}, fallback={}\n",
            self.stats.primary_rays_total,
            self.stats.primary_rays_accelerated,
            self.stats.primary_rays_fallback,
        );
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
