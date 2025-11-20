// src/scene/mod.rs

use std::rc::Rc;

// Point3, Color, Vec3, Sphere, HittableList, MaterialPtr, Lambertian, Metal, Dielectric, random_double, random_double_range
use crate::prelude::*; 

/// Más adelante, cuando se implemente la Tarea 8 (semillas),
/// este módulo scene es donde se inyectará un RNG con semilla
/// en lugar de usar random_double global.

/// Alias de conveniencia
pub type World = HittableList;

/// Tipo de escena que se quiere construir.
#[derive(Clone, Copy, Debug)]
pub enum SceneKind {
    /// La escena inicial sencilla
    Initial,
    /// La escena grande con muchas esferas aleatorias + 3 esferas grandes.
    ManySpheres,
}

/// Construye un World completo según el tipo de escena.
pub fn build_world(kind: SceneKind) -> World {
    let mut world = HittableList::new();

    match kind {
        SceneKind::Initial => build_initial_scene(&mut world),
        SceneKind::ManySpheres => build_many_spheres_scene(&mut world),
    }

    world
}

/// Escena inicial sencilla - Initial Scene
fn build_initial_scene(world: &mut HittableList) {
    let material_ground: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.8, 0.8, 0.0)));
    let material_center: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.1, 0.2, 0.5)));
    let material_left:   MaterialPtr = Rc::new(Dielectric::new(1.50));
    let material_bubble: MaterialPtr = Rc::new(Dielectric::new(1.0 / 1.5));
    let material_right:  MaterialPtr = Rc::new(Metal::new(Color::new(0.8, 0.6, 0.2), 1.0));

    world.add(Rc::new(Sphere::new(Point3::new( 0.0, -100.5, -1.0), 100.0, material_ground.clone())) as HittablePtr);
    world.add(Rc::new(Sphere::new(Point3::new( 0.0,    0.0, -1.2),   0.5, material_center.clone())) as HittablePtr);
    world.add(Rc::new(Sphere::new(Point3::new(-1.0,    0.0, -1.0),   0.5, material_left.clone()))   as HittablePtr);
    world.add(Rc::new(Sphere::new(Point3::new(-1.0,    0.0, -1.0),   0.4, material_bubble.clone())) as HittablePtr); // bubble
    world.add(Rc::new(Sphere::new(Point3::new( 1.0,    0.0, -1.0),   0.5, material_right.clone()))  as HittablePtr);
}

/// Escena grande con piso y muchas esferas aleatorias + 3 grandes.
fn build_many_spheres_scene(world: &mut HittableList) {
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
}
