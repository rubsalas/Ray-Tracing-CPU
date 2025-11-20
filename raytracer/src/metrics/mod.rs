// src/metrics/mod.rs

use std::time::Instant;


use crate::vec3::Color;
use crate::camera::Camera;
use crate::hittable::Hittable;
use crate::render::{RenderParams, BackendKind, Renderer};

pub mod runlog; // submódulo para escritura a archivo

/// Métricas de una sola ejecución de render.
#[derive(Debug)]
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
    // pub scene_seed: Option<u64>, // TODO
}

/// Encapsula la lógica de medición del render.
pub struct MetricsCollector;

impl MetricsCollector {
    /// Envuelve una llamada a `renderer.render(...)` y construye un RunMetrics.
    ///
    /// Recibe un trait object `&mut dyn Renderer`, que puede ser un ScalarRenderer,
    /// NeonRenderer u otro backend que implemente Renderer.
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

        let duration = start.elapsed();
        let image_height = camera.image_height();

        RunMetrics {
            run_id,
            backend,
            image_width: params.image_width,
            image_height,
            samples_per_pixel: params.samples_per_pixel,
            max_depth: params.max_depth,
            render_duration_ms: duration.as_millis(),
            cpu_user_ms: None,
            cpu_system_ms: None,
            peak_memory_bytes: None,
        }
    }
}

