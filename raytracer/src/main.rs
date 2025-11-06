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

use crate::prelude::*;
use crate::camera::Camera;
use crate::material::{Lambertian, Metal, MaterialPtr};
use crate::sphere::Sphere;
use crate::hittable_list::{HittableList, HittablePtr};

use std::rc::Rc;
use std::fs::File;
use std::io::{BufWriter, Result as IoResult};

fn main() -> IoResult<()> {
    // ---------- World ----------
    let mut world = HittableList::new();

    let material_ground: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.8, 0.8, 0.0)));
    let material_center: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.1, 0.2, 0.5)));
    let material_left:   MaterialPtr = Rc::new(Metal::new(Color::new(0.8, 0.8, 0.8)));
    let material_right:  MaterialPtr = Rc::new(Metal::new(Color::new(0.8, 0.6, 0.2)));

    world.add(Rc::new(Sphere::new(Point3::new( 0.0, -100.5, -1.0), 100.0, material_ground.clone())) as HittablePtr);
    world.add(Rc::new(Sphere::new(Point3::new( 0.0,    0.0, -1.2),   0.5, material_center.clone())) as HittablePtr);
    world.add(Rc::new(Sphere::new(Point3::new(-1.0,    0.0, -1.0),   0.5, material_left.clone())) as HittablePtr);
    world.add(Rc::new(Sphere::new(Point3::new( 1.0,    0.0, -1.0),   0.5, material_right.clone())) as HittablePtr);

    // ---------- Camera ----------
    let mut cam = Camera::new(400, 16.0 / 9.0);
    cam.samples_per_pixel = 100;
    cam.max_depth = 50;

    // ---------- Output ----------
    let file = File::create("image13.ppm")?;
    let mut out = BufWriter::new(file);

    cam.render(&world, &mut out)
}