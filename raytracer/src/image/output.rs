// src/image/output.rs

use std::path::Path;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write, Result as IoResult};

use crate::prelude::*; // Color, write_color_to, etc.
use crate::image::color::encode_color_scalar;

// NEON SIMD primitives are only available on aarch64 targets.
#[cfg(target_arch = "aarch64")]
use crate::simd::neon::F32x4;

/// Writes the framebuffer to a PPM (P3) file in ASCII format, using the
/// scalar post-processing path (`write_color_to` / `encode_color_scalar`).
///
/// This is the "reference" output path and is used:
/// - Always for the Scalar backend.
/// - As a fallback for the Neon backend on non-aarch64 targets.
pub fn write_ppm(
    path: &Path,
    image_width: i32,
    image_height: i32,
    framebuffer: &[Color],
) -> IoResult<()> {
    // Ensure the parent directory exists, if any.
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    // PPM header (P3 format).
    writeln!(out, "P3")?;
    writeln!(out, "{} {}", image_width, image_height)?;
    writeln!(out, "255")?;

    for j in 0..image_height {
        for i in 0..image_width {
            let idx = (j * image_width + i) as usize;
            let pixel_color = framebuffer[idx];
            // Scalar post-processing: gamma + clamp + quantize.
            write_color_to(&mut out, pixel_color)?;
        }
    }

    Ok(())
}

//
// ========== NEON helpers and SIMD-based PPM writer ==========
//

/// Clamps an `F32x4` to the range [`min_v`, `max_v`] lane-wise.
///
/// For each lane `i`:
///   result[i] = min(max(x[i], min_v), max_v).
#[cfg(target_arch = "aarch64")]
#[inline]
fn clamp_f32x4(x: F32x4, min_v: f32, max_v: f32) -> F32x4 {
    let lo = F32x4::splat(min_v);
    let hi = F32x4::splat(max_v);
    x.max(lo).min(hi)
}

/// Gamma-corrects four linear components at once using gamma = 2.0,
/// matching the scalar path (`linear_to_gamma` + sqrt).
#[cfg(target_arch = "aarch64")]
#[inline]
fn gamma_correct_f32x4(x: F32x4) -> F32x4 {
    // Gamma 2.0: sqrt(linear value).
    x.sqrt()
}

/// NEON-based encoder for a block of 4 linear colors.
///
/// Input:
/// - 4 `Color` values in linear space (f64, already scaled by 1/spp).
///
/// Output:
/// - 4 `(u8, u8, u8)` tuples ready to be written to a PPM file.
///
/// Steps (mirroring `encode_color_scalar` as closely as possible):
/// 1) Extract linear R,G,B components as `f32`.
/// 2) Pack them into three `F32x4` vectors (one per channel).
/// 3) Apply gamma correction with gamma = 2.0.
/// 4) Clamp each channel to [0.0, 0.999] in SIMD.
/// 5) Scale by 256 and convert to `u8` lane-wise.
#[cfg(target_arch = "aarch64")]
fn encode_color_block_neon(colors: [Color; 4]) -> [(u8, u8, u8); 4] {
    // 1) Extract components as f32 arrays.
    let mut rs = [0.0f32; 4];
    let mut gs = [0.0f32; 4];
    let mut bs = [0.0f32; 4];

    for i in 0..4 {
        rs[i] = colors[i].x as f32;
        gs[i] = colors[i].y as f32;
        bs[i] = colors[i].z as f32;
    }

    // 2) Pack into F32x4 vectors, one per channel.
    let r_vec = F32x4::from_array(rs);
    let g_vec = F32x4::from_array(gs);
    let b_vec = F32x4::from_array(bs);

    // 3) Gamma correction on all lanes.
    let r_gamma = gamma_correct_f32x4(r_vec);
    let g_gamma = gamma_correct_f32x4(g_vec);
    let b_gamma = gamma_correct_f32x4(b_vec);

    // 4) Clamp to [0.0, 0.999] in SIMD.
    let r_clamped = clamp_f32x4(r_gamma, 0.0, 0.999);
    let g_clamped = clamp_f32x4(g_gamma, 0.0, 0.999);
    let b_clamped = clamp_f32x4(b_gamma, 0.0, 0.999);

    // 5) Convert back to arrays and quantize to 0..255 as u8.
    let r_arr = r_clamped.to_array();
    let g_arr = g_clamped.to_array();
    let b_arr = b_clamped.to_array();

    let mut out = [(0u8, 0u8, 0u8); 4];
    for i in 0..4 {
        let r = (256.0 * r_arr[i]).floor() as u8;
        let g = (256.0 * g_arr[i]).floor() as u8;
        let b = (256.0 * b_arr[i]).floor() as u8;
        out[i] = (r, g, b);
    }

    out
}

/// Writes a PPM image using NEON SIMD for color post-processing (gamma + clamp)
/// in blocks of 4 pixels, and falls back to the scalar encoder for any "tail"
/// pixels at the end of each row.
///
/// Used when:
/// - Backend = Neon, and
/// - Target architecture = aarch64.
///
/// It is designed to produce output that is as close as possible to the
/// scalar `write_ppm` path.
#[cfg(target_arch = "aarch64")]
pub fn write_ppm_neon(
    path: &Path,
    image_width: i32,
    image_height: i32,
    framebuffer: &[Color],
) -> IoResult<()> {
    // Ensure the parent directory exists, same as in write_ppm.
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    // PPM header (P3 format).
    writeln!(out, "P3")?;
    writeln!(out, "{} {}", image_width, image_height)?;
    writeln!(out, "255")?;

    let width = image_width;
    let height = image_height;

    // Sanity check: framebuffer size must match width * height.
    assert_eq!(
        framebuffer.len(),
        (width * height) as usize,
        "framebuffer size mismatch in write_ppm_neon"
    );

    // Iterate over rows (scanlines).
    for j in 0..height {
        let row_start = j * width;
        let mut i = 0;

        // Process as many full 4-pixel blocks as possible using NEON.
        while i + 3 < width {
            let idx0 = (row_start + i) as usize;
            let idx1 = (row_start + i + 1) as usize;
            let idx2 = (row_start + i + 2) as usize;
            let idx3 = (row_start + i + 3) as usize;

            // Load 4 linear colors from the framebuffer.
            let block = [
                framebuffer[idx0],
                framebuffer[idx1],
                framebuffer[idx2],
                framebuffer[idx3],
            ];

            // Encode the 4 pixels in parallel (gamma + clamp + quantize).
            let encoded = encode_color_block_neon(block);

            // Write each encoded pixel as a PPM line "R G B".
            for (r, g, b) in encoded.iter() {
                writeln!(out, "{} {} {}", r, g, b)?;
            }

            i += 4;
        }

        // Tail: if the row width is not a multiple of 4, there can be
        // 1 to 3 remaining pixels. We process those using the scalar
        // reference encoder to keep the logic simple and safe.
        while i < width {
            let idx = (row_start + i) as usize;
            let (r, g, b) = encode_color_scalar(framebuffer[idx]);
            writeln!(out, "{} {} {}", r, g, b)?;
            i += 1;
        }
    }

    Ok(())
}

// Tests específicos para el postprocesado NEON vs escalar.
// Solo se compilan/ejecutan en aarch64, donde existe F32x4.
#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::image::color::encode_color_scalar;
    use crate::prelude::Color;

    /// Helper: returns true if the difference between two u8
    /// values is at most 1. We allow a tolerance of 1 to account
    /// for tiny differences between f32 and f64 computations.
    fn close_u8(a: u8, b: u8) -> bool {
        let da = a as i16;
        let db = b as i16;
        (da - db).abs() <= 1
    }

    #[test]
    fn encode_color_block_neon_matches_scalar_for_basic_colors() {
        // Four representative colors in linear space [0,1].
        let c0 = Color::new(0.0, 0.0, 0.0);
        let c1 = Color::new(0.25, 0.5, 0.75);
        let c2 = Color::new(0.5, 0.5, 0.5);
        let c3 = Color::new(1.0, 1.0, 1.0);

        let colors = [c0, c1, c2, c3];

        // Scalar reference: encode each color independently.
        let mut scalar_encoded = [(0u8, 0u8, 0u8); 4];
        for i in 0..4 {
            scalar_encoded[i] = encode_color_scalar(colors[i]);
        }

        // NEON version: encode the four colors in one block.
        let neon_encoded = encode_color_block_neon(colors);

        // Compare lane by lane, allowing a difference of at most 1
        // per channel to account for f32 vs f64 rounding.
        for i in 0..4 {
            let (sr, sg, sb) = scalar_encoded[i];
            let (nr, ng, nb) = neon_encoded[i];

            assert!(
                close_u8(sr, nr),
                "Red channel mismatch at lane {}: scalar={} neon={}",
                i,
                sr,
                nr
            );
            assert!(
                close_u8(sg, ng),
                "Green channel mismatch at lane {}: scalar={} neon={}",
                i,
                sg,
                ng
            );
            assert!(
                close_u8(sb, nb),
                "Blue channel mismatch at lane {}: scalar={} neon={}",
                i,
                sb,
                nb
            );
        }
    }
}
