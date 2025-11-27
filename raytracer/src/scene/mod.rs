// src/scene/mod.rs

use std::rc::Rc;

use crate::prelude::*;
use crate::world::hittable_list::{HittableList, HittablePtr};
use crate::world::sphere::Sphere;

/// Alias de conveniencia para el mundo de objetos.
pub type World = HittableList;

/// Tipo de escena que se quiere construir.
#[derive(Clone, Copy, Debug)]
pub enum SceneKind {
    /// La escena inicial sencilla.
    Initial,
    /// La escena grande con muchas esferas aleatorias + 3 esferas grandes.
    ManySpheres,
    // A futuro se podría agregar aquí:
    // RandomSmall,
    // CornellBox,
    // etc.
}

/// Datos completos de una escena:
/// - `world`: lista de objetos utilizada por el camino escalar (HittableList).
/// - `spheres`: copia plana de las esferas, usada para aceleración NEON.
pub struct SceneData {
    pub world: World,
    pub spheres: Vec<Sphere>,
}

/// Construye una escena completa (World + Vec<Sphere>) según el tipo.
///
/// Esta es la función “nueva” que usará:
/// - El backend escalar → `SceneData.world`.
/// - El backend NEON → `SceneData.spheres` (para world_hit4_spheres).
pub fn build_scene(kind: SceneKind) -> SceneData {
    let mut world = HittableList::new();
    let mut spheres: Vec<Sphere> = Vec::new();

    match kind {
        SceneKind::Initial => build_initial_scene(&mut world, &mut spheres),
        SceneKind::ManySpheres => build_many_spheres_scene(&mut world, &mut spheres),
    }

    SceneData { world, spheres }
}

/// Construye solo el `World` escalar como antes.
///
/// Esto mantiene compatibilidad con cualquier código que siga llamando
/// `build_world(SceneKind)`. Por dentro simplemente delega a `build_scene`.
pub fn build_world(kind: SceneKind) -> World {
    build_scene(kind).world
}

/// Escena inicial sencilla - Initial Scene.
///
/// Rellena:
/// - `world`: HittableList usado por el raytracer.
/// - `spheres`: copia plana de las esferas para la aceleración NEON.
fn build_initial_scene(world: &mut HittableList, spheres: &mut Vec<Sphere>) {
    let material_ground: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.8, 0.8, 0.0)));
    let material_center: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.1, 0.2, 0.5)));
    let material_left:   MaterialPtr = Rc::new(Dielectric::new(1.50));
    let material_bubble: MaterialPtr = Rc::new(Dielectric::new(1.0 / 1.5));
    let material_right:  MaterialPtr = Rc::new(Metal::new(Color::new(0.8, 0.6, 0.2), 1.0));

    // Ground
    let s_ground = Sphere::new(
        Point3::new( 0.0, -100.5, -1.0),
        100.0,
        material_ground,
    );
    spheres.push(s_ground.clone());
    world.add(Rc::new(s_ground) as HittablePtr);

    // Center
    let s_center = Sphere::new(
        Point3::new( 0.0, 0.0, -2.0),
        0.5,
        material_center,
    );
    spheres.push(s_center.clone());
    world.add(Rc::new(s_center) as HittablePtr);

    // Left (outer)
    let s_left = Sphere::new(
        Point3::new(-1.0, 0.0, -1.0),
        0.5,
        material_left,
    );
    spheres.push(s_left.clone());
    world.add(Rc::new(s_left) as HittablePtr);

    // Bubble (inner)
    let s_bubble = Sphere::new(
        Point3::new(-1.0, 0.0, -1.0),
        0.4,
        material_bubble,
    );
    spheres.push(s_bubble.clone());
    world.add(Rc::new(s_bubble) as HittablePtr);

    // Right
    let s_right = Sphere::new(
        Point3::new( 1.0, 0.0, 1.0),
        0.5,
        material_right,
    );
    spheres.push(s_right.clone());
    world.add(Rc::new(s_right) as HittablePtr);
}

/// Escena grande con piso y muchas esferas aleatorias + 3 grandes.
///
/// Igual que antes, pero ahora:
/// - Cada Sphere que se agrega al `world` también se guarda en `spheres`.
fn build_many_spheres_scene(world: &mut HittableList, spheres: &mut Vec<Sphere>) {
    // Ground
    let ground_material: MaterialPtr =
        Rc::new(Lambertian::new(Color::new(0.5, 0.5, 0.5)));
    let s_ground = Sphere::new(
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        ground_material,
    );
    spheres.push(s_ground.clone());
    world.add(Rc::new(s_ground) as HittablePtr);

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
                if choose_mat < 0.8 {
                    // Diffuse
                    let albedo = Color::random() * Color::random();
                    let sphere_material: MaterialPtr =
                        Rc::new(Lambertian::new(albedo));
                    let s = Sphere::new(center, 0.2, sphere_material);
                    spheres.push(s.clone());
                    world.add(Rc::new(s) as HittablePtr);

                } else if choose_mat < 0.95 {
                    // Metal
                    let albedo = Color::random_range(0.5, 1.0);
                    let fuzz   = random_double_range(0.0, 0.5);
                    let sphere_material: MaterialPtr =
                        Rc::new(Metal::new(albedo, fuzz));
                    let s = Sphere::new(center, 0.2, sphere_material);
                    spheres.push(s.clone());
                    world.add(Rc::new(s) as HittablePtr);

                } else {
                    // Glass
                    let sphere_material: MaterialPtr = Rc::new(Dielectric::new(1.5));
                    let s = Sphere::new(center, 0.2, sphere_material);
                    spheres.push(s.clone());
                    world.add(Rc::new(s) as HittablePtr);
                }
            }
        }
    }

    // Three big spheres
    let material1: MaterialPtr = Rc::new(Dielectric::new(1.5));
    let s1 = Sphere::new(Point3::new( 0.0, 1.0, 0.0), 1.0, material1);
    spheres.push(s1.clone());
    world.add(Rc::new(s1) as HittablePtr);

    let material2: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.4, 0.2, 0.1)));
    let s2 = Sphere::new(Point3::new(-4.0, 1.0, 0.0), 1.0, material2);
    spheres.push(s2.clone());
    world.add(Rc::new(s2) as HittablePtr);

    let material3: MaterialPtr = Rc::new(Metal::new(Color::new(0.7, 0.6, 0.5), 0.0));
    let s3 = Sphere::new(Point3::new( 4.0, 1.0, 0.0), 1.0, material3);
    spheres.push(s3.clone());
    world.add(Rc::new(s3) as HittablePtr);
}
