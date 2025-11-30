// src/image/output.rs

use std::path::Path;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write, Result as IoResult};

use crate::prelude::*;
use crate::image::color::{write_color_to, encode_color_scalar};

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
    // Se asegura que el directorio padre exista.
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    // Encabezado PPM (formato P3).
    writeln!(out, "P3")?;
    writeln!(out, "{} {}", image_width, image_height)?;
    writeln!(out, "255")?;

    // Orden de recorrido: j = 0..image_height, i = 0..image_width
    // Índice lineal: idx = j * width + i
    for j in 0..image_height {
        for i in 0..image_width {
            let idx = (j * image_width + i) as usize;
            let pixel_color = framebuffer[idx];
            // Postprocesado escalar: gamma + clamp + cuantización.
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
/// Para cada lane `i`:
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
    // Gamma 2.0: sqrt(valor lineal).
    x.sqrt()
}

/// NEON-based encoder for a block of 4 linear colors.
///
/// Entrada:
/// - 4 `Color` en espacio lineal (f64, ya escalados por 1/spp).
///
/// Salida:
/// - 4 tuplas `(u8, u8, u8)` listas para escribir en el PPM.
///
/// Pasos (imitando `encode_color_scalar`):
/// 1) Extraer R,G,B lineales como `f32`.
/// 2) Empaquetar en tres `F32x4` (un canal por vector).
/// 3) Aplicar corrección gamma 2.0.
/// 4) Clampear cada canal a [0.0, 0.999].
/// 5) Multiplicar por 256 y convertir a `u8`.
#[cfg(target_arch = "aarch64")]
fn encode_color_block_neon(colors: [Color; 4]) -> [(u8, u8, u8); 4] {
    // 1) Extraer componentes como f32.
    let mut rs = [0.0f32; 4];
    let mut gs = [0.0f32; 4];
    let mut bs = [0.0f32; 4];

    for i in 0..4 {
        rs[i] = colors[i].x as f32;
        gs[i] = colors[i].y as f32;
        bs[i] = colors[i].z as f32;
    }

    // 2) Empaquetar en F32x4 por canal.
    let r_vec = F32x4::from_array(rs);
    let g_vec = F32x4::from_array(gs);
    let b_vec = F32x4::from_array(bs);

    // 3) Gamma en todos los lanes.
    let r_gamma = gamma_correct_f32x4(r_vec);
    let g_gamma = gamma_correct_f32x4(g_vec);
    let b_gamma = gamma_correct_f32x4(b_vec);

    // 4) Clamp a [0.0, 0.999].
    let r_clamped = clamp_f32x4(r_gamma, 0.0, 0.999);
    let g_clamped = clamp_f32x4(g_gamma, 0.0, 0.999);
    let b_clamped = clamp_f32x4(b_gamma, 0.0, 0.999);

    // 5) Volver a arrays y cuantizar a 0..255.
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

/// PPM writer usando NEON para el postprocesado (gamma + clamp) en bloques
/// de 4 píxeles, con cola escalar para los píxeles restantes.
///
/// IMPORTANTE: recorre la imagen en el **mismo orden** que `write_ppm`:
/// - `j` de 0 a `image_height - 1`
/// - `i` de 0 a `image_width - 1`
/// - índice lineal: `idx = j * image_width + i_lane`
#[cfg(target_arch = "aarch64")]
pub fn write_ppm_neon(
    path: &Path,
    image_width: i32,
    image_height: i32,
    framebuffer: &[Color],
) -> IoResult<()> {
    // Se asegura que el directorio padre exista.
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    // Encabezado PPM (P3).
    writeln!(out, "P3")?;
    writeln!(out, "{} {}", image_width, image_height)?;
    writeln!(out, "255")?;

    let width = image_width;
    let height = image_height;

    assert_eq!(
        framebuffer.len(),
        (width * height) as usize,
        "framebuffer size mismatch in write_ppm_neon"
    );

    // Mismo patrón de recorrido que el writer escalar.
    for j in 0..height {
        let mut i = 0;

        // Bloques de 4 píxeles con NEON.
        while i + 3 < width {
            let idx0 = (j * width + i) as usize;
            let idx1 = (j * width + i + 1) as usize;
            let idx2 = (j * width + i + 2) as usize;
            let idx3 = (j * width + i + 3) as usize;

            let block = [
                framebuffer[idx0],
                framebuffer[idx1],
                framebuffer[idx2],
                framebuffer[idx3],
            ];

            let encoded = encode_color_block_neon(block);

            for (r, g, b) in encoded.iter() {
                writeln!(out, "{} {} {}", r, g, b)?;
            }

            i += 4;
        }

        // Cola 1–3 píxeles con encoder escalar.
        while i < width {
            let idx = (j * width + i) as usize;
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

    /// Se considera que dos u8 son equivalentes si difieren a lo sumo en 1.
    /// Esto permite pequeñas diferencias de redondeo f32 vs f64.
    fn close_u8(a: u8, b: u8) -> bool {
        let da = a as i16;
        let db = b as i16;
        (da - db).abs() <= 1
    }

    #[test]
    fn encode_color_block_neon_matches_scalar_for_basic_colors() {
        let c0 = Color::new(0.0, 0.0, 0.0);
        let c1 = Color::new(0.25, 0.5, 0.75);
        let c2 = Color::new(0.5, 0.5, 0.5);
        let c3 = Color::new(1.0, 1.0, 1.0);

        let colors = [c0, c1, c2, c3];

        let mut scalar_encoded = [(0u8, 0u8, 0u8); 4];
        for i in 0..4 {
            scalar_encoded[i] = encode_color_scalar(colors[i]);
        }

        let neon_encoded = encode_color_block_neon(colors);

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
