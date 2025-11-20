// src/output.rs

use std::path::Path;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write, Result as IoResult};

use crate::prelude::*;

// Escribe un framebuffer en formato PPM (P3) en el archivo indicado.
pub fn write_ppm(
    path: &Path,
    image_width: i32,
    image_height: i32,
    framebuffer: &[Color],
) -> IoResult<()> {
    // Crear directorio padre si existe
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    writeln!(out, "P3")?;
    writeln!(out, "{} {}", image_width, image_height)?;
    writeln!(out, "255")?;

    for j in 0..image_height {
        for i in 0..image_width {
            let idx = (j * image_width + i) as usize;
            let pixel_color = framebuffer[idx];
            write_color_to(&mut out, pixel_color)?;
        }
    }

    Ok(())
}
