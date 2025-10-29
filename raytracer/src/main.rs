mod vec3;
mod color;
mod ray;

mod hittable;
mod sphere;
mod hittable_list;

use sphere::Sphere;
use hittable_list::{HittableList, HittablePtr};

use std::fs::File;
use std::io::{self, BufWriter, Write};
use vec3::{Vec3, Point3, Color, dot, unit_vector};
use color::write_color_to;
use ray::Ray;

/// Intersección rayo–esfera (forma "half-b").
/// Ecuación de la esfera: |O + t·d - C|^2 = R^2
/// Sea oc = O - C.
/// Expandiendo: (d·d) t^2 + 2(oc·d) t + (oc·oc - R^2) = 0
///
/// Definimos:
///   a       = d·d
///   half_b  = oc·d            (la mitad de "b" original)
///   c       = oc·oc - R^2
/// Entonces:
///   discriminant = half_b^2 - a*c
///
/// Si discriminant < 0 → no hay raíces reales → no hay impacto.
/// Si ≥ 0 → hay dos raíces; la más cercana es t = (-half_b - sqrt(discriminant)) / a
///
/// Devuelve Some(t) si hay impacto, o None si no. `t` es el parámetro sobre el rayo.
fn hit_sphere(center: Point3, radius: f64, r: Ray) -> Option<f64> {
    // oc = center - origin
    let oc = center - r.origin();

    // Forma simplificada
    let a = r.direction().length_squared();           // a = |d|^2
    let h = dot(r.direction(), oc);                   // h = d · oc
    let c = oc.length_squared() - radius * radius;    // c = |oc|^2 - r^2

    let discriminant = h * h - a * c;
    if discriminant < 0.0 {
        None
    } else {
        // t más cercano hacia adelante
        Some((h - discriminant.sqrt()) / a)
    }
}

/// Colorea por normal de superficie cuando hay impacto; si no, dibuja el cielo.
///
/// - Si el rayo golpea la esfera, calculamos la normal `n` en el punto de impacto:
///     n = unit_vector(P(t) - C)
///   Luego mapeamos de [-1,1] a [0,1] para visualizar la normal: 0.5*(n + 1).
///
/// - Si no hay impacto, usamos el gradiente de cielo del libro.
fn ray_color(r: Ray) -> Color {
    if let Some(t) = hit_sphere(Point3::new(0.0, 0.0, -1.0), 0.5, r) {
        let hit_point = r.at(t);
        let n = unit_vector(hit_point - Point3::new(0.0, 0.0, -1.0));

        // Remapeo [-1,1] → [0,1]:  0.5*(n + 1)
        return 0.5 * Vec3::new(n.x + 1.0, n.y + 1.0, n.z + 1.0);
    }

    // Cielo (gradiente) si no hay impacto
    let d = unit_vector(r.direction());
    let a = 0.5 * (d.y + 1.0);
    (1.0 - a) * Vec3::new(1.0, 1.0, 1.0) + a * Vec3::new(0.5, 0.7, 1.0)
}

fn main() -> io::Result<()> {
    // ---------------- Image ----------------
    let aspect_ratio: f64 = 16.0 / 9.0;
    let image_width: i32 = 400;
    let mut image_height: i32 = (image_width as f64 / aspect_ratio) as i32;
    if image_height < 1 { image_height = 1; }

    // ---------------- Camera ----------------
    let camera_center: Point3 = Vec3::new(0.0, 0.0, 0.0);
    let focal_length: f64 = 1.0;
    let viewport_height: f64 = 2.0;
    let viewport_width: f64 = viewport_height * (image_width as f64 / image_height as f64);

    let viewport_u: Vec3 = Vec3::new(viewport_width, 0.0, 0.0);
    let viewport_v: Vec3 = Vec3::new(0.0, -viewport_height, 0.0); // hacia abajo

    let pixel_delta_u: Vec3 = viewport_u / (image_width as f64);
    let pixel_delta_v: Vec3 = viewport_v / (image_height as f64);

    let viewport_upper_left: Point3 =
        camera_center - Vec3::new(0.0, 0.0, focal_length) - (viewport_u / 2.0) - (viewport_v / 2.0);

    let pixel00_loc: Point3 = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

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

            let pixel_color = ray_color(r);
            write_color_to(&mut out, pixel_color)?;
        }
    }

    out.flush()?;
    eprintln!("\rDone.                 ");
    Ok(())
}
