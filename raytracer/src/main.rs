mod vec3;
mod color;
mod ray;
mod interval;
mod hittable;
mod hittable_list;
mod material;
mod sphere;
mod prelude;
mod camera;
mod render;
mod output;

use crate::prelude::*;
use crate::camera::Camera;
use crate::render::{RenderParams, BackendKind, make_renderer};
use crate::output::write_ppm;

use std::rc::Rc;
use std::path::Path;
use std::io::{Result as IoResult};

fn main() -> IoResult<()> {
    // ---------- World ----------
    let mut world = HittableList::new();

    /* Initial Scene */
    // let material_ground: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.8, 0.8, 0.0)));
    // let material_center: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.1, 0.2, 0.5)));
    // let material_left:   MaterialPtr = Rc::new(Dielectric::new(1.50));
    // let material_bubble: MaterialPtr = Rc::new(Dielectric::new(1.0 / 1.5));
    // let material_right:  MaterialPtr = Rc::new(Metal::new(Color::new(0.8, 0.6, 0.2), 1.0));

    // world.add(Rc::new(Sphere::new(Point3::new( 0.0, -100.5, -1.0), 100.0, material_ground.clone())) as HittablePtr);
    // world.add(Rc::new(Sphere::new(Point3::new( 0.0,    0.0, -1.2),   0.5, material_center.clone())) as HittablePtr);
    // world.add(Rc::new(Sphere::new(Point3::new(-1.0,    0.0, -1.0),   0.5, material_left.clone()))   as HittablePtr);
    // world.add(Rc::new(Sphere::new(Point3::new(-1.0,    0.0, -1.0),   0.4, material_bubble.clone())) as HittablePtr); // bubble
    // world.add(Rc::new(Sphere::new(Point3::new( 1.0,    0.0, -1.0),   0.5, material_right.clone()))  as HittablePtr);

    // Ground
    let ground_material: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.5, 0.5, 0.5)));
    world.add(Rc::new(Sphere::new(Point3::new(0.0, -1000.0, 0.0), 1000.0, ground_material)) as HittablePtr);

    // Small random spheres grid
    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random_double(); // [0,1)
            let center = Point3::new(
                a as f64 + 0.9 * random_double(),
                0.2,
                b as f64 + 0.9 * random_double(),
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                // Diffuse
                if choose_mat < 0.8 {
                    // albedo = random() * random()  (element-wise for darker tones)
                    let albedo = Color::random() * Color::random();
                    let sphere_material: MaterialPtr = Rc::new(Lambertian::new(albedo));
                    world.add(Rc::new(Sphere::new(center, 0.2, sphere_material)) as HittablePtr);

                // Metal
                } else if choose_mat < 0.95 {
                    let albedo = Color::random_range(0.5, 1.0);
                    let fuzz   = random_double_range(0.0, 0.5);
                    let sphere_material: MaterialPtr = Rc::new(Metal::new(albedo, fuzz));
                    world.add(Rc::new(Sphere::new(center, 0.2, sphere_material)) as HittablePtr);

                // Glass
                } else {
                    let sphere_material: MaterialPtr = Rc::new(Dielectric::new(1.5));
                    world.add(Rc::new(Sphere::new(center, 0.2, sphere_material)) as HittablePtr);
                }
            }
        }
    }

    // Three big spheres
    let material1: MaterialPtr = Rc::new(Dielectric::new(1.5));
    world.add(Rc::new(Sphere::new(Point3::new( 0.0, 1.0, 0.0), 1.0, material1)) as HittablePtr);

    let material2: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.4, 0.2, 0.1)));
    world.add(Rc::new(Sphere::new(Point3::new(-4.0, 1.0, 0.0), 1.0, material2)) as HittablePtr);

    let material3: MaterialPtr = Rc::new(Metal::new(Color::new(0.7, 0.6, 0.5), 0.0));
    world.add(Rc::new(Sphere::new(Point3::new( 4.0, 1.0, 0.0), 1.0, material3)) as HittablePtr);

    // ---------- Camera ----------
    let mut cam = Camera::new(/*1200*/ 400, 16.0 / 9.0);
    cam.samples_per_pixel = 100; /*500*/
    cam.max_depth = 50;

    cam.vfov = 20.0;
    cam.lookfrom = Point3::new(13.0, 2.0, 3.0);
    cam.lookat   = Point3::new( 0.0, 0.0, 0.0);
    cam.vup      = Vec3::new(0.0, 1.0, 0.0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    // ---------- Render Params ----------
    let params = RenderParams {
        image_width: cam.image_width,
        aspect_ratio: cam.aspect_ratio,
        samples_per_pixel: cam.samples_per_pixel,
        max_depth: cam.max_depth,
    };

    // Altura deducida igual que en camera.initialize
    let mut image_height = (params.image_width as f64 / params.aspect_ratio) as i32;
    if image_height < 1 { image_height = 1; }

    // Framebuffer en memoria: un Color por píxel
    let mut framebuffer = vec![
        Color::new(0.0, 0.0, 0.0);
        (params.image_width * image_height) as usize
    ];

    // ---------- Render backend ----------
    let backend_kind = BackendKind::Scalar; // luego se podrá cambiar a Neon
    let mut renderer = make_renderer(backend_kind);

    renderer.render(&world, &mut cam, &params, &mut framebuffer);

    // ---------- Output ----------
    // Nombre de archivo según backend o config
    let output_path = Path::new("output/image_scalar.ppm");
    write_ppm(output_path, params.image_width, image_height, &framebuffer)?;

    Ok(())
}
