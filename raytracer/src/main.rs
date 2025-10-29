mod vec3;
mod color;
mod ray;

mod hittable;
mod hittable_list;
mod sphere;

mod prelude;

mod interval;

// un único use que trae constantes, helpers y tipos:
// (pi, infinity, degrees_to_radians, Vec3, Point3, Color, Ray,
//  write_color_to, Hittable, HitRecord, Sphere, HittableList, HittablePtr, etc.)
use crate::prelude::*;

use std::fs::File;
use std::io::{self, BufWriter, Write};

/// Background + surface color
/// - Si hay hit: 0.5 * (normal + (1,1,1))  → normal en [0,1].
/// - Si no hay hit: gradiente vertical (blanco → azul).
fn ray_color(r: &Ray, world: &impl Hittable) -> Color {
    let mut rec = HitRecord::default();
    if world.hit(r, &Interval::new(0.0, INFINITY), &mut rec) {
        // 0.5 * (n + 1)
        return 0.5 * (rec.normal + Color::new(1.0, 1.0, 1.0));
    }

    let unit_direction = unit_vector(r.direction());
    let a = 0.5 * (unit_direction.y + 1.0);
    (1.0 - a) * Color::new(1.0, 1.0, 1.0) + a * Color::new(0.5, 0.7, 1.0)
}

fn main() -> io::Result<()> {
    // ---------------- Image ----------------
    let aspect_ratio: f64 = 16.0 / 9.0;
    let image_width: i32 = 720;
    
    // Calcular height y asegurar al menos 1
    let mut image_height: i32 = (image_width as f64 / aspect_ratio) as i32;
    if image_height < 1 {
        image_height = 1;
    }

    // -------------
    // World (2 esferas)
    // -------------
    let mut world = HittableList::new();
    world.add(std::rc::Rc::new(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5)) as HittablePtr);
    world.add(std::rc::Rc::new(Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0)) as HittablePtr);

    // -------------
    // Camera (pinhole)
    // -------------
    let focal_length = 1.0;
    let viewport_height = 2.0;
    let viewport_width = viewport_height * (image_width as f64 / image_height as f64);
    let camera_center = Point3::new(0.0, 0.0, 0.0);

    // Bordes del viewport en el espacio de cámara
    let viewport_u = Vec3::new(viewport_width, 0.0, 0.0);
    let viewport_v = Vec3::new(0.0, -viewport_height, 0.0);

    // Paso entre píxeles en u y v
    let pixel_delta_u = viewport_u / image_width as f64;
    let pixel_delta_v = viewport_v / image_height as f64;

    // Coordenada del pixel (0,0): esquina sup-izq desplazada a centro del píxel
    let viewport_upper_left =
        camera_center - Vec3::new(0.0, 0.0, focal_length) - viewport_u / 2.0 - viewport_v / 2.0;
    let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // -------------- Render a archivo --------------
    let file = File::create("image.ppm")?;
    let mut out = BufWriter::new(file);

    writeln!(out, "P3")?;
    writeln!(out, "{} {}", image_width, image_height)?;
    writeln!(out, "255")?;

    for j in 0..image_height {
        eprint!("\rScanlines remaining: {} ", image_height - j);
        io::stderr().flush().ok();

        for i in 0..image_width {
            let pixel_center = pixel00_loc + (i as f64) * pixel_delta_u + (j as f64) * pixel_delta_v;
            let ray_direction = pixel_center - camera_center;
            let r = Ray::new(camera_center, ray_direction);

            let pixel_color = ray_color(&r, &world);
            write_color_to(&mut out, pixel_color)?;
        }
    }

    out.flush()?;
    eprintln!("\rDone.                 ");
    Ok(())
}
