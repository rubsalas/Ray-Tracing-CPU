// src/scene/mod.rs

use std::rc::Rc;

use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::prelude::*;
use crate::world::hittable_list::{HittableList, HittablePtr};
use crate::world::sphere::Sphere;

/// Alias de conveniencia para el tipo de mundo.
pub type World = HittableList;

/// Tipo de escena que se quiere construir.
#[derive(Clone, Copy, Debug)]
pub enum SceneKind {
    /// Escena inicial sencilla.
    Initial,
    /// Escena grande con muchas esferas aleatorias + 3 esferas grandes.
    ManySpheres,
}

/// Datos completos que describe una escena:
/// - `world`: lista de objetos hittable (para el camino escalar).
/// - `spheres`: copia plana de esferas (para el backend NEON y world_hit4_spheres).
pub struct SceneData {
    pub world: World,
    pub spheres: Vec<Sphere>,
}

/// Construye la escena completa (World + Vec<Sphere]) según el tipo y la semilla.
///
/// `scene_seed` permite fijar la aleatoriedad:
/// - `Some(seed)` -> escena determinista.
/// - `None`      -> escena con aleatoriedad no determinista (se usa from_entropy()).
pub fn build_scene(kind: SceneKind, scene_seed: Option<u64>) -> SceneData {
    let mut world = HittableList::new();
    let mut spheres_for_accel: Vec<Sphere> = Vec::new();

    match kind {
        SceneKind::Initial => {
            build_initial_scene(&mut world, &mut spheres_for_accel);
        }
        SceneKind::ManySpheres => {
            build_many_spheres_scene(&mut world, &mut spheres_for_accel, scene_seed);
        }
    }

    SceneData {
        world,
        spheres: spheres_for_accel,
    }
}

/// Helper interno para registrar una esfera tanto en el mundo hittable
/// como en el vector plano de esferas para NEON.
///
/// Se construye una instancia para el mundo y otra para el vector plano,
/// compartiendo el mismo `MaterialPtr` (Rc).
fn add_sphere_to_world_and_list(
    world: &mut HittableList,
    spheres: &mut Vec<Sphere>,
    center: Point3,
    radius: f64,
    material: &MaterialPtr,
) {
    // Se crea la esfera para el mundo (envuelta en Rc y HittablePtr).
    let world_sphere = Sphere::new(center, radius, material.clone());
    world.add(Rc::new(world_sphere) as HittablePtr);

    // Se crea la esfera para el vector plano (se comparte el mismo material Rc).
    let accel_sphere = Sphere::new(center, radius, material.clone());
    spheres.push(accel_sphere);
}

/// Escena inicial sencilla - Initial Scene
fn build_initial_scene(world: &mut HittableList, spheres: &mut Vec<Sphere>) {
    let material_ground: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.8, 0.8, 0.0)));
    let material_center: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.1, 0.2, 0.5)));
    let material_left:   MaterialPtr = Rc::new(Dielectric::new(1.50));
    let material_bubble: MaterialPtr = Rc::new(Dielectric::new(1.0 / 1.5));
    let material_right:  MaterialPtr = Rc::new(Metal::new(Color::new(0.8, 0.6, 0.2), 1.0));

    add_sphere_to_world_and_list(
        world,
        spheres,
        Point3::new(0.0, -100.5, -1.0),
        100.0,
        &material_ground,
    );

    add_sphere_to_world_and_list(
        world,
        spheres,
        Point3::new(0.0, 0.0, -2.0),
        0.5,
        &material_center,
    );

    add_sphere_to_world_and_list(
        world,
        spheres,
        Point3::new(-1.0, 0.0, -1.0),
        0.5,
        &material_left,
    );

    // Esfera interna tipo "burbuja".
    add_sphere_to_world_and_list(
        world,
        spheres,
        Point3::new(-1.0, 0.0, -1.0),
        0.4,
        &material_bubble,
    );

    add_sphere_to_world_and_list(
        world,
        spheres,
        Point3::new(1.0, 0.0, 1.0),
        0.5,
        &material_right,
    );
}

/// Genera un color aleatorio con componentes en [0, 1) usando el RNG dado.
fn random_color_unit(rng: &mut StdRng) -> Color {
    let dist = Uniform::new(0.0_f64, 1.0_f64);
    let r = rng.sample(dist);
    let g = rng.sample(dist);
    let b = rng.sample(dist);
    Color::new(r, g, b)
}

/// Genera un color aleatorio con componentes en [min, max) usando el RNG dado.
fn random_color_range(rng: &mut StdRng, min: f64, max: f64) -> Color {
    let dist = Uniform::new(min, max);
    let r = rng.sample(dist);
    let g = rng.sample(dist);
    let b = rng.sample(dist);
    Color::new(r, g, b)
}

/// Escena grande con piso y muchas esferas aleatorias + 3 grandes.
///
/// Se usa `StdRng` sembrado con `scene_seed` para que las posiciones y materiales
/// aleatorios se mantengan deterministas siempre que la semilla sea la misma.
fn build_many_spheres_scene(
    world: &mut HittableList,
    spheres: &mut Vec<Sphere>,
    scene_seed: Option<u64>,
) {
    // Se construye el RNG determinista (o desde entropía si no hay semilla).
    let mut rng: StdRng = match scene_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let dist01 = Uniform::new(0.0_f64, 1.0_f64);
    let dist_fuzz = Uniform::new(0.0_f64, 0.5_f64);

    // --------- Ground ---------
    let ground_material: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.5, 0.5, 0.5)));
    add_sphere_to_world_and_list(
        world,
        spheres,
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        &ground_material,
    );

    // --------- Small random spheres grid ---------
    for a in -11..11 {
        for b in -11..11 {
            let choose_mat: f64 = rng.sample(dist01);
            let center = Point3::new(
                a as f64 + 0.9 * rng.sample(dist01),
                0.2,
                b as f64 + 0.9 * rng.sample(dist01),
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                // Diffuse
                if choose_mat < 0.8 {
                    // albedo = random() * random()  (más oscuro)
                    let albedo = random_color_unit(&mut rng) * random_color_unit(&mut rng);
                    let sphere_material: MaterialPtr = Rc::new(Lambertian::new(albedo));

                    add_sphere_to_world_and_list(
                        world,
                        spheres,
                        center,
                        0.2,
                        &sphere_material,
                    );

                // Metal
                } else if choose_mat < 0.95 {
                    let albedo = random_color_range(&mut rng, 0.5, 1.0);
                    let fuzz: f64 = rng.sample(dist_fuzz);
                    let sphere_material: MaterialPtr = Rc::new(Metal::new(albedo, fuzz));

                    add_sphere_to_world_and_list(
                        world,
                        spheres,
                        center,
                        0.2,
                        &sphere_material,
                    );

                // Glass
                } else {
                    let sphere_material: MaterialPtr = Rc::new(Dielectric::new(1.5));
                    add_sphere_to_world_and_list(
                        world,
                        spheres,
                        center,
                        0.2,
                        &sphere_material,
                    );
                }
            }
        }
    }

    // --------- Three big spheres ---------
    let material1: MaterialPtr = Rc::new(Dielectric::new(1.5));
    add_sphere_to_world_and_list(
        world,
        spheres,
        Point3::new(0.0, 1.0, 0.0),
        1.0,
        &material1,
    );

    let material2: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.4, 0.2, 0.1)));
    add_sphere_to_world_and_list(
        world,
        spheres,
        Point3::new(-4.0, 1.0, 0.0),
        1.0,
        &material2,
    );

    let material3: MaterialPtr = Rc::new(Metal::new(Color::new(0.7, 0.6, 0.5), 0.0));
    add_sphere_to_world_and_list(
        world,
        spheres,
        Point3::new(4.0, 1.0, 0.0),
        1.0,
        &material3,
    );
}
