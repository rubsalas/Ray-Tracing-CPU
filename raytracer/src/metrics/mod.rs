// src/metrics/mod.rs

use std::time::Instant;

use crate::vec3::Color;
use crate::camera::Camera;
use crate::world::hittable::Hittable;
use crate::render::neon::NeonRenderer;
use crate::render::{RenderParams, BackendKind, Renderer};

use self::core_stats::{CoreStats, reset_core_stats, snapshot_core_stats};

pub mod core_stats;
pub mod runlog; // submódulo para escritura a archivo

/// Métricas de una sola ejecución de render.
pub struct RunMetrics {
    pub run_id: String,
    pub backend: BackendKind,
    pub image_width: i32,
    pub image_height: i32,
    pub samples_per_pixel: i32,
    pub max_depth: i32,

    pub scene_seed: Option<u64>,
    
    pub render_duration_ms: u128,
    pub cpu_user_ms: Option<u128>,
    pub cpu_system_ms: Option<u128>,
    pub peak_memory_bytes: Option<u64>,

    // Contadores específicos del backend NEON.
    // Para el backend escalar se mantienen en 0.
    pub primary_rays_total: u64,
    pub primary_rays_accelerated: u64,
    pub primary_rays_fallback: u64,

    // Contadores lógicos compartidos por todos los backends.
    pub core_stats: CoreStats,
}

impl RunMetrics {
    pub fn new(
        run_id: String,
        backend: BackendKind,
        image_width: i32,
        image_height: i32,
        samples_per_pixel: i32,
        max_depth: i32,
        render_duration_ms: u128,
    ) -> Self {
        Self {
            run_id,
            backend,
            image_width,
            image_height,
            samples_per_pixel,
            max_depth,
            scene_seed: None,

            render_duration_ms,
            cpu_user_ms: None,
            cpu_system_ms: None,
            peak_memory_bytes: None,

            // Por defecto, counters en 0.
            primary_rays_total: 0,
            primary_rays_accelerated: 0,
            primary_rays_fallback: 0,

            // NUEVO: stats lógicos por defecto.
            core_stats: CoreStats::default(),
        }
    }
}


/// Tipo marcador para agrupar funciones relacionadas con la medición
/// de métricas de ejecución.
pub struct MetricsCollector;

/// Encapsula la lógica de medición del render.
impl MetricsCollector {
    pub fn measure_render(
        renderer: &mut dyn Renderer,
        backend: BackendKind,
        run_id: String,
        params: &RenderParams,
        world: &dyn Hittable,
        camera: &mut Camera,
        framebuffer: &mut [Color],
    ) -> RunMetrics {
        // Se resetean los contadores lógicos antes del render.
        reset_core_stats();

        let image_width = params.image_width;
        let mut image_height = camera.image_height();
        if image_height <= 0 {
            // Se asegura que la altura se mantenga consistente.
            image_height =
                (image_width as f64 / params.aspect_ratio) as i32;
            if image_height < 1 {
                image_height = 1;
            }
        }

        let start = Instant::now();

        renderer.render(world, camera, params, framebuffer);

        let elapsed = start.elapsed();
        let render_duration_ms = elapsed.as_millis();

        // Se obtienen contadores lógicos acumulados.
        let core_stats = snapshot_core_stats();

        // Se obtienen contadores específicos de NEON (si aplica).
        let mut primary_rays_total = 0_u64;
        let mut primary_rays_accelerated = 0_u64;
        let mut primary_rays_fallback = 0_u64;

        if let BackendKind::Neon = backend {
            if let Some(neon) = renderer
                .as_any()
                .downcast_ref::<crate::render::NeonRenderer>()
            {
                let s = neon.stats;
                primary_rays_total = s.primary_rays_total;
                primary_rays_accelerated = s.primary_rays_accelerated;
                primary_rays_fallback = s.primary_rays_fallback;
            }
        }

        RunMetrics {
            run_id,
            backend,
            image_width,
            image_height,
            samples_per_pixel: params.samples_per_pixel,
            max_depth: params.max_depth,

            // Se rellena en execute_run con config.scene_seed.
            scene_seed: None,

            render_duration_ms,
            cpu_user_ms: None,
            cpu_system_ms: None,
            peak_memory_bytes: None,

            primary_rays_total,
            primary_rays_accelerated,
            primary_rays_fallback,

            core_stats,
        }
    }
}


