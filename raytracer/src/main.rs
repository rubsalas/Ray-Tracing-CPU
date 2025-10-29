mod vec3;
mod color;
mod ray;
mod hittable;
mod sphere;
mod hittable_list;
mod interval;
mod prelude;
mod camera;

use crate::prelude::*;
use crate::camera::Camera;
use std::fs::File;
use std::io::{BufWriter, Result as IoResult};

fn main() -> IoResult<()> {
    // Build the world
    let mut world = HittableList::new();
    world.add(std::rc::Rc::new(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5)) as HittablePtr);
    world.add(std::rc::Rc::new(Sphere::new(Point3::new(0.0, -100.5, -1.0), 100.0)) as HittablePtr);

    // Camera
    let mut cam = Camera::new(400, 16.0 / 9.0);

    // Output
    let file = File::create("image.ppm")?;
    let mut out = BufWriter::new(file);

    // Render
    cam.render(&world, &mut out)
}
