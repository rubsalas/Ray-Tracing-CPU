// src/render/primary.rs

//! Helpers for primary-ray shading that can reuse SIMD-accelerated
//! intersection information, but keep the shading logic scalar and
//! as close as possible to the original book implementation.

use crate::prelude::*;
use crate::camera::Camera;
use crate::world::hittable::{Hittable, HitRecord};
use crate::world::sphere::Sphere;
use crate::interval::Interval;

/// Computes the background color for a ray, matching the logic used
/// in `Camera::ray_color` when there is no hit in the scene.
///
/// This is the classic gradient from the book:
/// white at the bottom to blue-ish at the top.
fn background_color_for_ray(r: &Ray) -> Color {
    let unit_direction = unit_vector(r.direction());
    let a = 0.5 * (unit_direction.y + 1.0);
    (1.0 - a) * Color::new(1.0, 1.0, 1.0)
        + a * Color::new(0.5, 0.7, 1.0)
}

/// Computes the color for a *primary* ray, optionally using a known
/// sphere index (obtained via SIMD / HitInfo4) to avoid doing a full
/// `world.hit` scan for the first bounce.
///
/// Parameters:
/// - `camera`: the camera, used for recursive calls to `ray_color`.
/// - `world`: the full world (HittableList) for secondary bounces.
/// - `ray`: the primary ray [camera → scene].
/// - `depth`: remaining recursion depth for this ray.
/// - `has_hit`: true if SIMD said this lane hit some sphere.
/// - `sphere_index`: index into `spheres` for the hit sphere; must be
///   valid iff `has_hit == true`.
/// - `spheres`: raw vector of spheres corresponding to the world.
///
/// Behavior:
/// - If `depth <= 0`: returns black (no more light gathered), same as
///   `Camera::ray_color`.
/// - If `has_hit == false`:
///     - Returns the background gradient color (no `world.hit`).
/// - If `has_hit == true`:
///     - Calls `spheres[sphere_index].hit(ray, ...)` once to build a
///       scalar `HitRecord` for this primary bounce.
///     - If, por razones numéricas, ese `hit` falla, hace fallback a
///       `camera.ray_color(ray, depth, world)` para no romper nada.
///     - Si hay hit y hay material:
///         - Llama a `mat.scatter(...)` igual que en `Camera::ray_color`.
///         - Llama recursivamente a `camera.ray_color` para el rayo
///           secundario, decreciendo `depth`.
///       Si el material absorbe: retorna negro.
pub fn shade_primary_with_sphere_hint(
    camera: &Camera,
    world: &dyn Hittable,
    ray: &Ray,
    depth: i32,
    has_hit: bool,
    sphere_index: i32,
    spheres: &[Sphere],
) -> Color {
    // 1) Límite de rebotes (igual que en Camera::ray_color).
    if depth <= 0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    // 2) Caso "no hit" según la información SIMD:
    //    devolvemos el fondo directamente, sin llamar a world.hit.
    if !has_hit || sphere_index < 0 {
        return background_color_for_ray(ray);
    }

    // 3) Tenemos índice de esfera válido: usamos esa sphere concreta.
    let idx = sphere_index as usize;
    if idx >= spheres.len() {
        // Índice inválido (no debería pasar si HitInfo4 está bien)
        // pero mejor hacer fallback seguro.
        return camera.ray_color(ray, depth, world);
    }

    let sphere = &spheres[idx];

    // Intervalo de t igual al usado en Camera::ray_color (0.001..∞).
    let t_min = 0.001_f64;
    let t_max = INFINITY;
    let interval = Interval::new(t_min, t_max);

    let mut rec = HitRecord::default();

    // 4) Usamos la intersección escalar de esta sphere concreta.
    if !sphere.hit(ray, &interval, &mut rec) {
        // Por estabilidad numérica, podría fallar aunque SIMD dijera hit;
        // en ese caso, no arriesgamos cambios de comportamiento y
        // delegamos en la implementación escalar completa.
        return camera.ray_color(ray, depth, world);
    }

    // 5) Si hay material, seguimos exactamente la lógica de la versión
    //    escalar para el primer rebote.
    if let Some(mat) = &rec.mat {
        let mut scattered = Ray::new(rec.p, Vec3::new(0.0, 0.0, 0.0));
        let mut attenuation = Color::new(0.0, 0.0, 0.0);

        if mat.scatter(ray, &rec, &mut attenuation, &mut scattered) {
            // Para los rebotes siguientes usamos Camera::ray_color normal,
            // que seguirá usando world.hit (escalar) para todos los rebotes
            // posteriores.
            return attenuation * camera.ray_color(&scattered, depth - 1, world);
        }

        // Material que absorbe completamente en este rebote.
        return Color::new(0.0, 0.0, 0.0);
    }

    // 6) Si por alguna razón no hay material asociado, consideramos
    //    que no hay contribución (igual que en la rama de absorción).
    Color::new(0.0, 0.0, 0.0)
}
