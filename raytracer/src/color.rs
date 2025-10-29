//! `color` module
//!
//! Helpers to format linear RGB colors as ASCII PPM (`P3`) lines. A color is
//! represented by the semantic alias [`Color`] (equal to [`Vec3`](crate::vec3::Vec3))
//! where `x=r`, `y=g`, `z=b`.
//!
//! At this stage we assume linear inputs in `[0.0, 1.0]` per channel and
//! perform a simple clamp-and-quantize to 8-bit. Gamma correction and sampling
//! averages come later in the book.
//!
//! # Example
//! ```rust,no_run
//! # use crate::vec3::Vec3;
//! # use crate::color::write_color_to;
//! let c = Vec3::new(0.2, 0.6, 1.0);
//! let mut buf = Vec::new();
//! write_color_to(&mut buf, c).unwrap();
//! assert_eq!(std::str::from_utf8(&buf).unwrap().trim(), "51 153 255");
//! ```

use std::io::{Result as IoResult, Write};
use crate::vec3::Color;

/// Clamps `x` to the closed interval `[min, max]`.
///
/// Used to keep channels within displayable range before quantizing to `0..=255`.
#[inline]
fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min { min } else if x > max { max } else { x }
}

/// Writes `"R G B\n"` to any `Write` sink (file, buffer, stdout) in ASCII PPM style.
///
/// The input is expected to be linear RGB with each component in `[0.0, 1.0]`.
/// Channels are clamped to that range, scaled by `255.999`, and truncated to `u32`
/// so that an exact `1.0` maps to `255`.
///
/// # Errors
/// Propagates any I/O error from the underlying writer.
///
/// # Example
/// ```rust,no_run
/// # use crate::vec3::Vec3;
/// # use crate::color::write_color_to;
/// let mut out = Vec::new();
/// write_color_to(&mut out, Vec3::new(0.0, 0.5, 1.0)).unwrap();
/// assert_eq!(std::str::from_utf8(&out).unwrap(), "0 127 255\n");
/// ```
pub fn write_color_to<W: Write>(out: &mut W, pixel_color: Color) -> IoResult<()> {
    let r = pixel_color.x;
    let g = pixel_color.y;
    let b = pixel_color.z;

    // Map [0,1] -> [0,255] using 255.999 so that 1.0 becomes 255.
    let rbyte = (255.999 * clamp(r, 0.0, 1.0)) as u32;
    let gbyte = (255.999 * clamp(g, 0.0, 1.0)) as u32;
    let bbyte = (255.999 * clamp(b, 0.0, 1.0)) as u32;

    writeln!(out, "{rbyte} {gbyte} {bbyte}")
}
