// src/render/mod.rs

use crate::prelude::*;
use crate::camera::Camera;
use crate::hittable::Hittable;

/// Parámetros lógicos del render
pub struct RenderParams {
    pub image_width: i32,
    pub aspect_ratio: f64,
    pub samples_per_pixel: i32,
    pub max_depth: i32,
}

#[derive(Clone, Copy, Debug)]
pub enum BackendKind {
    Scalar,
    Neon,
}

/// Interfaz común para los backends de render.
pub trait Renderer {
    fn render(
        &mut self,
        world: &dyn Hittable,
        camera: &mut Camera,
        params: &RenderParams,
        framebuffer: &mut [Color],
    );
}

mod scalar;
mod neon;

pub use scalar::ScalarRenderer;
pub use neon::NeonRenderer;

/// Fábrica de backends según el enum.
pub fn make_renderer(kind: BackendKind) -> Box<dyn Renderer> {
    match kind {
        BackendKind::Scalar => Box::new(ScalarRenderer::new()),
        BackendKind::Neon   => Box::new(NeonRenderer::new()),
    }
}
