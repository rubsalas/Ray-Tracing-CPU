// src/render/neon.rs

use crate::prelude::*;
use crate::camera::Camera;
use crate::hittable::Hittable;

use super::{RenderParams, Renderer};

use super::ScalarRenderer;

pub struct NeonRenderer {
    scalar_fallback: ScalarRenderer,
}

impl NeonRenderer {
    pub fn new() -> Self {
        NeonRenderer {
            scalar_fallback: ScalarRenderer::new(),
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
        // Tarea 0: delegar al backend escalar.
        // Luego se ira cambiando esto por la versión SIMD.
        self.scalar_fallback
            .render(world, camera, params, framebuffer);
    }
}
