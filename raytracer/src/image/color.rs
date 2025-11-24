//! `color` module — (gamma 2.0)
//!
//! Formats linear RGB colors as ASCII PPM (`P3`) lines.
//! This version applies a linear to gamma transform
//! with gamma = 2.0 (`sqrt`) before clamping and quantizing to 8-bit.

use std::io::{Result as IoResult, Write};

use crate::interval::Interval;
use crate::vec3::Color;

/// Linear to gamma mapping for gamma = 2.0.
///
/// Returns `sqrt(linear_component)` for positive inputs, otherwise `0.0`.
#[inline]
pub fn linear_to_gamma(linear_component: f64) -> f64 {
    if linear_component > 0.0 {
        linear_component.sqrt()
    } else {
        0.0
    }
}

/// Scalar reference encoder: converts a linear `Color` into
/// 8-bit RGB components ready to be written to a PPM file.
///
/// Steps (matching the original implementation):
/// 1) Read linear RGB (expected in `[0, 1]` per channel).
/// 2) Apply gamma-2.0 correction via [`linear_to_gamma`].
/// 3) Clamp each channel to `[0.000, 0.999]` using [`Interval`].
/// 4) Map to bytes using `int(256 * value)`.
pub fn encode_color_scalar(pixel_color: Color) -> (u8, u8, u8) {
    // 1) Linear RGB in [0,1]
    let mut r = pixel_color.x;
    let mut g = pixel_color.y;
    let mut b = pixel_color.z;

    // 2) Gamma 2.0
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // 3) Clamp to [0.000, 0.999]
    let intensity = Interval::new(0.0, 0.999);

    // 4) Quantize to bytes with int(256 * value)
    let rbyte = (256.0 * intensity.clamp(r)) as u32;
    let gbyte = (256.0 * intensity.clamp(g)) as u32;
    let bbyte = (256.0 * intensity.clamp(b)) as u32;

    // We return u8 because the PPM writer and the NEON path
    // both work naturally with 0..=255 values. Casting from
    // u32 to u8 is safe here because the range is bounded.
    (rbyte as u8, gbyte as u8, bbyte as u8)
}

/// Writes `"R G B\n"` to any `Write` sink (file, buffer, stdout) in ASCII PPM style.
///
/// This is now just a thin wrapper around [`encode_color_scalar`],
/// so all the post-processing logic is centralized in one place.
/// The NEON path will reuse the same semantics for its SIMD encoder.
pub fn write_color_to<W: Write>(out: &mut W, pixel_color: Color) -> IoResult<()> {
    let (r, g, b) = encode_color_scalar(pixel_color);
    writeln!(out, "{} {} {}", r, g, b)
}
