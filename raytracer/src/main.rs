// src/main.rs
mod vec3;
mod color;

use std::fs::File;
use std::io::{self, BufWriter, Write};
use vec3::{Vec3, Color};
use color::write_color_to;

fn main() -> io::Result<()> {
    // Image
    let image_width: u32 = 256;
    let image_height: u32 = 256;

    // Render: abrimos archivo y escribimos encabezado PPM (P3)
    let file = File::create("image.ppm")?;
    let mut out = BufWriter::new(file);

    writeln!(out, "P3")?;
    writeln!(out, "{} {}", image_width, image_height)?;
    writeln!(out, "255")?;

    for j in 0..image_height {
        // Progreso por stderr (equivalente a std::clog)
        eprint!("\rScanlines remaining: {} ", image_height - j);
        io::stderr().flush().ok();

        for i in 0..image_width {
            // color(double(i)/(image_width-1), double(j)/(image_height-1), 0)
            let pixel_color: Color = Vec3::new(
                (i as f64) / ((image_width - 1) as f64),
                (j as f64) / ((image_height - 1) as f64),
                0.0,
            );

            // write_color(out, pixel_color)
            write_color_to(&mut out, pixel_color)?;
        }
    }

    // Asegura que el buffer se vacíe al archivo
    out.flush()?;

    eprintln!("\rDone.                 ");
    Ok(())
}
