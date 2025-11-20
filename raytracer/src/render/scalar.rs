// src/render/scalar.rs

use crate::prelude::*;
use crate::camera::Camera;
use crate::world::hittable::Hittable;

use super::{RenderParams, Renderer};

pub struct ScalarRenderer;

impl ScalarRenderer {
    pub fn new() -> Self {
        ScalarRenderer
    }
}

impl Renderer for ScalarRenderer {
    fn render(
        &mut self,
        world: &dyn Hittable,
        camera: &mut Camera,
        params: &RenderParams,
        framebuffer: &mut [Color],
    ) {
        // Prepara la cámara (igual que antes en Camera::render)
        camera.initialize();

        let image_width = params.image_width;
        let image_height = camera.image_height(); // calculado en initialize
        let samples_per_pixel = params.samples_per_pixel;
        let max_depth = params.max_depth;
        let pixel_scale = camera.pixel_samples_scale();

        assert_eq!(
            framebuffer.len(),
            (image_width * image_height) as usize,
            "framebuffer size mismatch"
        );

        for j in 0..image_height {
            eprint!("\rScanlines remaining: {} ", image_height - j);
            use std::io::Write as _;
            std::io::stderr().flush().ok();

            for i in 0..image_width {
                // Sum of samples for this pixel
                let mut pixel_color = Color::new(0.0, 0.0, 0.0);

                for _s in 0..samples_per_pixel {
                    let r = camera.get_ray(i, j);
                    pixel_color += camera.ray_color(&r, max_depth, world);
                }

                let idx = (j * image_width + i) as usize;
                // Aquí ya aplicamos el 1/spp
                framebuffer[idx] = pixel_scale * pixel_color;
            }
        }

        eprintln!("\rDone.                 ");
    }
}
