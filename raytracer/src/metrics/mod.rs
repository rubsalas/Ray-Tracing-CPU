// src/metrics/mod.rs

use std::time::Instant;

use crate::vec3::Color;
use crate::camera::Camera;
use crate::world::hittable::Hittable;
use crate::render::neon::NeonRenderer;
use crate::render::{RenderParams, BackendKind, Renderer};

pub mod runlog; // submódulo para escritura a archivo

/// Métricas de una sola ejecución de render.
pub struct RunMetrics {
    pub run_id: String,
    pub backend: BackendKind,
    pub image_width: i32,
    pub image_height: i32,
    pub samples_per_pixel: i32,
    pub max_depth: i32,
    pub render_duration_ms: u128,

    pub cpu_user_ms: Option<u128>,
    pub cpu_system_ms: Option<u128>,
    pub peak_memory_bytes: Option<u64>,

    // NUEVO: contadores específicos del backend NEON.
    // Para el backend escalar se quedarán en 0.
    pub primary_rays_total: u64,
    pub primary_rays_accelerated: u64,
    pub primary_rays_fallback: u64,

    // pub scene_seed: Option<u64>, // TODO
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
            render_duration_ms,

            cpu_user_ms: None,
            cpu_system_ms: None,
            peak_memory_bytes: None,

            // Por defecto, counters en 0.
            primary_rays_total: 0,
            primary_rays_accelerated: 0,
            primary_rays_fallback: 0,
        }
    }
}

/// Encapsula la lógica de medición del render.
pub struct MetricsCollector;

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
        let start = Instant::now();

        renderer.render(world, camera, params, framebuffer);

        let duration_ms = start.elapsed().as_millis();

        let mut metrics = RunMetrics::new(
            run_id,
            backend,
            params.image_width,
            camera.image_height(),
            params.samples_per_pixel,
            params.max_depth,
            duration_ms,
        );

        // Aquí, más adelante, se podrá rellenar cpu_user_ms, etc.

        // NUEVO: si el backend es NEON, se lee NeonStats y se guardan.
        if let BackendKind::Neon = backend {
            if let Some(neon) = renderer.as_any().downcast_ref::<NeonRenderer>() {
                metrics.primary_rays_total       = neon.stats.primary_rays_total;
                metrics.primary_rays_accelerated = neon.stats.primary_rays_accelerated;
                metrics.primary_rays_fallback    = neon.stats.primary_rays_fallback;
            }
        }

        metrics
    }
}

