// src/main.rs

mod vec3;
mod color;
mod ray;

use std::fs::File;
use std::io::{self, BufWriter, Write};

use vec3::{Vec3, Point3, Color, unit_vector};
use color::write_color_to;
use ray::Ray;


fn ray_color(r: Ray) -> Color {
    // Normaliza la dirección del rayo (para mapear el cielo por dirección)
    let unit_direction = unit_vector(r.direction());

    // 'a' va de 0..1 según el componente Y del rayo (de -1..1 a 0..1)
    let a = 0.5 * (unit_direction.y + 1.0);
    // Mezcla lineal: (1-a)*blanco + a*azul-cielo
    (1.0 - a) * Vec3::new(1.0, 1.0, 1.0) + a * Vec3::new(0.5, 0.7, 1.0)
}

fn main() -> io::Result<()> {
    // ------------------------------------------------------------------------
    // Image
    // ------------------------------------------------------------------------
    let aspect_ratio: f64 = 16.0 / 9.0;
    let image_width: i32 = 400;

    // Calcula el alto y garantiza que sea al menos 1 (igual que en C++).
    let mut image_height: i32 = (image_width as f64 / aspect_ratio) as i32;
    if image_height < 1 {
        image_height = 1;
    }

    // ------------------------------------------------------------------------
    // Camera (pinhole)
    // ------------------------------------------------------------------------
    let camera_center: Point3 = Vec3::new(0.0, 0.0, 0.0);

    let focal_length: f64 = 1.0;
    let viewport_height: f64 = 2.0;
    let viewport_width: f64 = viewport_height * (image_width as f64 / image_height as f64);

    // Vectores a lo largo de los bordes del viewport (horizontal y vertical).
    // v vertical apunta hacia abajo (y negativa) para que el origen (0,0) quede arriba-izq.
    let viewport_u: Vec3 = Vec3::new(viewport_width, 0.0, 0.0);
    let viewport_v: Vec3 = Vec3::new(0.0, -viewport_height, 0.0);

    // Deltas por píxel (horizontal y vertical).
    let pixel_delta_u: Vec3 = viewport_u / (image_width as f64);
    let pixel_delta_v: Vec3 = viewport_v / (image_height as f64);

    // Ubicación de la esquina superior izquierda del viewport.
    let viewport_upper_left: Point3 =
        camera_center - Vec3::new(0.0, 0.0, focal_length) - (viewport_u / 2.0) - (viewport_v / 2.0);

    // Centro del píxel (0,0): esquina sup-izq + 0.5*(delta_u + delta_v)
    let pixel00_loc: Point3 = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // ------------------------------------------------------------------------
    // Render (a archivo) + progreso por stderr
    // ------------------------------------------------------------------------
    let file = File::create("image.ppm")?;
    let mut out = BufWriter::new(file);

    // Encabezado PPM (P3)
    writeln!(out, "P3")?;
    writeln!(out, "{} {}", image_width, image_height)?;
    writeln!(out, "255")?;

    for j in 0..image_height {
        // Progreso (stderr)
        eprint!("\rScanlines remaining: {} ", image_height - j);
        io::stderr().flush().ok();

        for i in 0..image_width {
            // Centro del píxel (i,j)
            let pixel_center: Point3 =
                pixel00_loc + (i as f64) * pixel_delta_u + (j as f64) * pixel_delta_v;

            // Dirección del rayo desde el centro de la cámara al centro del píxel
            let ray_direction: Vec3 = pixel_center - camera_center;

            // Rayo primario
            let r = Ray::new(camera_center, ray_direction);

            // Color del rayo
            let pixel_color: Color = ray_color(r);

            // Escribir "R G B\n" al archivo
            write_color_to(&mut out, pixel_color)?;
        }
    }

    out.flush()?;
    eprintln!("\rDone.                 ");

    Ok(())
}
