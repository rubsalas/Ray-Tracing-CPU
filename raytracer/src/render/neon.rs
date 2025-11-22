// src/render/neon.rs

use crate::prelude::*;
use crate::camera::Camera;
use crate::world::hittable::Hittable;

use super::{RenderParams, Renderer};

/// Backend NEON en primera etapa:
/// - Recorre la imagen por bloques de 4 píxeles en X.
/// - Dentro de cada bloque, calcula cada píxel usando el mismo
///   camino totalmente escalar que ScalarRenderer.
/// 
/// Más adelante, estos bloques de 4 serán reemplazados por
/// operaciones SIMD reales con core::arch::aarch64.
pub struct NeonRenderer;

impl NeonRenderer {
    pub fn new() -> Self {
        NeonRenderer
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
        // Igual que en ScalarRenderer: preparar la cámara.
        camera.initialize();

        let image_width = params.image_width;
        let image_height = camera.image_height();
        let samples_per_pixel = params.samples_per_pixel;
        let max_depth = params.max_depth;
        let pixel_scale = camera.pixel_samples_scale();

        assert_eq!(
            framebuffer.len(),
            (image_width * image_height) as usize,
            "framebuffer size mismatch"
        );

        for j in 0..image_height {
            // Mensaje de progreso, igual que el backend escalar.
            eprint!("\rScanlines remaining (NEON): {} ", image_height - j);
            use std::io::Write as _;
            std::io::stderr().flush().ok();

            let row_start = j * image_width;

            // Recorremos la fila en bloques de 4 píxeles.
            for i in (0..image_width).step_by(4) {
                // Cuántos píxeles reales hay en este bloque.
                // En la mayoría de casos será 4, salvo al final de la fila
                // si image_width no es múltiplo de 4.
                let remaining = image_width - i;
                let count = remaining.min(4);

                for lane in 0..count {
                    let ix = i + lane;

                    // Color acumulado para el píxel (ix, j)
                    let mut pixel_color = Color::new(0.0, 0.0, 0.0);

                    for _s in 0..samples_per_pixel {
                        let r = camera.get_ray(ix, j);
                        pixel_color += camera.ray_color(&r, max_depth, world);
                    }

                    let idx = (row_start + ix) as usize;
                    framebuffer[idx] = pixel_scale * pixel_color;
                }
            }
        }

        eprintln!("\rDone (NEON).           ");
    }
}
