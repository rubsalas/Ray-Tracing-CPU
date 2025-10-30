//! `color` module — Listing 44 port
//!
//! Formats linear RGB colors as ASCII PPM (`P3`) lines. This version follows
//! RTIOW Listing 44 by clamping each channel with an [`Interval`] of
//! `[0.000, 0.999]` and mapping with `int(256 * value)`.
//!
//! A color is represented by the alias [`Color`] (equals [`Vec3`](crate::vec3::Vec3)):
//! `x=r`, `y=g`, `z=b`.

use std::io::{Result as IoResult, Write};

use crate::interval::Interval;
use crate::vec3::Color;

/// Writes `"R G B\n"` to any `Write` sink (file, buffer, stdout) in ASCII PPM style.
///
/// Each channel is assumed **linear** in `[0.0, 1.0]`. Per Listing 44,
/// values are clamped to `[0.000, 0.999]` and converted with `int(256 * value)`
/// so that the top code point never rounds up to 256.
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
/// assert_eq!(std::str::from_utf8(&out).unwrap(), "0 128 255\n");
/// ```
pub fn write_color_to<W: Write>(out: &mut W, pixel_color: Color) -> IoResult<()> {
    let r = pixel_color.x;
    let g = pixel_color.y;
    let b = pixel_color.z;

    // Listing 44: clamp to [0.000, 0.999] then scale by 256.
    let intensity = Interval::new(0.0, 0.999);

    let rbyte = (256.0 * intensity.clamp(r)) as u32;
    let gbyte = (256.0 * intensity.clamp(g)) as u32;
    let bbyte = (256.0 * intensity.clamp(b)) as u32;

    writeln!(out, "{rbyte} {gbyte} {bbyte}")
}
