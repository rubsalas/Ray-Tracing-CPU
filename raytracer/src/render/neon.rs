// src/render/neon.rs

use crate::prelude::*;
use crate::camera::Camera;
use crate::world::hittable::Hittable;

use super::{RenderParams, Renderer};
use super::ScalarRenderer;

#[cfg(target_arch = "aarch64")]
use crate::simd::neon::Ray4;

/// NEON backend for the renderer.
///
/// In phase 4.1, this backend:
/// - Iterates pixels in blocks of 4 horizontally.
/// - For each block, builds 4 rays using the scalar Camera::get_ray
///   via Camera::get_ray4_from_indices (Ray4 as a container).
/// - Still calls ray_color scalar per lane.
/// - Accumulates samples per pixel exactly like the scalar backend.
///
/// Later phases will move more math (ray construction, shading, etc.)
/// into NEON, but this step ensures Ray4 is correctly integrated.
pub struct NeonRenderer {
    /// A scalar backend that can be used as a fallback if needed.
    /// In this phase we don't call it directly inside render(), but
    /// we keep it around in case we need a pure scalar path.
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

    /// NEON-based renderer backend.
    ///
    /// Current state (phase 4):
    /// - Iterates horizontally in blocks of 4 pixels.
    /// - Uses `Camera::get_ray4_from_indices(i, j)` to build 4 rays in parallel
    ///   using NEON (`F32x4` + `Vec3x4`) for the camera geometry.
    /// - Unpacks each lane from `Ray4` back into a scalar `Ray` and calls the
    ///   existing `ray_color` implementation (still fully scalar).
    /// - Accumulates samples per pixel exactly like the scalar backend and
    ///   writes the linear color into the framebuffer.
    ///
    /// Only the camera stage (ray construction) and the color postprocessing
    /// (in the image/output module) make use of NEON at this point. Intersection,
    /// material evaluation and recursion (`ray_color`) are still scalar, which
    /// makes it easier to validate correctness against the reference path.
    fn render(
        &mut self,
        world: &dyn Hittable,
        camera: &mut Camera,
        params: &RenderParams,
        framebuffer: &mut [Color],
    ) {
        // Prepare camera (same as scalar renderer).
        camera.initialize();

        let image_width = params.image_width;
        let image_height = camera.image_height();
        let samples_per_pixel = params.samples_per_pixel;
        let max_depth = params.max_depth;
        let pixel_scale = camera.pixel_samples_scale();

        assert_eq!(
            framebuffer.len(),
            (image_width * image_height) as usize,
            "framebuffer size mismatch in NeonRenderer"
        );

        for j in 0..image_height {
            eprint!("\r[NEON] Scanlines remaining: {} ", image_height - j);
            use std::io::Write as _;
            std::io::stderr().flush().ok();

            let mut i = 0;

            // --- Main loop: process as many full blocks of 4 pixels as possible ---
            #[cfg(target_arch = "aarch64")]
            {
                while i + 3 < image_width {
                    // Accumulated linear color per lane (before 1/spp).
                    let mut lane_colors = [
                        Color::new(0.0, 0.0, 0.0),
                        Color::new(0.0, 0.0, 0.0),
                        Color::new(0.0, 0.0, 0.0),
                        Color::new(0.0, 0.0, 0.0),
                    ];

                    // Multi-sampling loop.
                    for _s in 0..samples_per_pixel {
                        // Build four rays at once for pixels (i, i+1, i+2, i+3) at row j.
                        // Internally this still calls Camera::get_ray scalar four times,
                        // but the result is packed in a Ray4.
                        let ray4 = camera.get_ray4_from_indices(i, j);

                        // Extract each lane as a scalar Ray and shade it via ray_color.
                        for lane in 0..4 {
                            let ray_lane = ray4.lane(lane);
                            lane_colors[lane] += camera.ray_color(&ray_lane, max_depth, world);
                        }
                    }

                    // Store the averaged color into the framebuffer for each pixel in the block.
                    for lane in 0..4 {
                        let x = i + lane as i32;
                        let idx = (j * image_width + x) as usize;
                        framebuffer[idx] = pixel_scale * lane_colors[lane];
                    }

                    i += 4;
                }
            }

            // --- Tail: remaining pixels at the end of the row (0 to 3 pixels) ---
            //
            // We keep a scalar path that mimics ScalarRenderer's inner loop,
            // so we don't need to handle partial Ray4 blocks in this phase.
            while i < image_width {
                let mut pixel_color = Color::new(0.0, 0.0, 0.0);

                for _s in 0..samples_per_pixel {
                    let r = camera.get_ray(i, j);
                    pixel_color += camera.ray_color(&r, max_depth, world);
                }

                let idx = (j * image_width + i) as usize;
                framebuffer[idx] = pixel_scale * pixel_color;

                i += 1;
            }
        }

        eprintln!("\r[NEON] Done.                 ");
    }
}
