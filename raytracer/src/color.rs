// src/color.rs
// Utilidades para escribir un Color (Vec3) en un archivo PPM.

use std::io::{Result as IoResult, Write};
use crate::vec3::Color;

/// clamp: limita x al rango [min, max].
#[inline]
fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min { min } else if x > max { max } else { x }
}

/// Escribe "R G B\n" a cualquier destino que implemente `Write` (archivo, buffer, stdout).
/// `pixel_color` debe venir en el rango [0,1] por canal.
pub fn write_color_to<W: Write>(out: &mut W, pixel_color: Color) -> IoResult<()> {
    let r = pixel_color.x;
    let g = pixel_color.y;
    let b = pixel_color.z;

    // Mapea [0,1] -> [0,255] (usa 255.999 para que 1.0 llegue a 255).
    let rbyte = (255.999 * clamp(r, 0.0, 1.0)) as u32;
    let gbyte = (255.999 * clamp(g, 0.0, 1.0)) as u32;
    let bbyte = (255.999 * clamp(b, 0.0, 1.0)) as u32;

    writeln!(out, "{rbyte} {gbyte} {bbyte}")
}
