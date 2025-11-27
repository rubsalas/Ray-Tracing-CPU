// src/simd/hit.rs

//! SIMD hit utilities
//!
//! This module provides SIMD-friendly helpers for ray–geometry intersection.
//! In this phase we focus on sphere intersection for four rays in parallel,
//! returning per-lane hit information in a compact struct.

use crate::simd::neon::{F32x4, Vec3x4, Ray4};
use crate::world::sphere::Sphere;

/// Per-lane hit information for a batch of four rays.
///
/// Each lane corresponds to one ray in a `Ray4`. For each lane we store:
/// - `hit`: whether the ray actually intersects the sphere within [t_min, t_max].
/// - `t`: the chosen intersection distance along the ray (in world units).
/// - `sphere_index`: the logical index of the sphere in the world (or -1 if no hit).
///
/// In this phase we keep `t` and `hit` as scalar arrays for easier integration
/// with the existing scalar pipeline and tests. Later we could introduce
/// fully SIMD masks if needed.
pub struct HitInfo4 {
    /// For each lane (0..4), true if that ray hit the sphere/world.
    pub hit: [bool; 4],

    /// Parametric t value per lane (válido solo si `hit[lane] == true`).
    pub t: [f32; 4],

    /// Sphere index per lane (–1 si no hay hit en ese rayo).
    pub sphere_index: [i32; 4],
}

impl HitInfo4 {
    /// Helper to construct a "no hit" result with all `t` initialized to `initial_t`
    /// and all indices set to -1.
    pub fn no_hit(initial_t: f32) -> Self {
        HitInfo4 {
            hit: [false; 4],
            t: [initial_t; 4],
            sphere_index: [-1; 4],
        }
    }
}

/// Computes the intersection of four rays with a single sphere in parallel.
///
/// For each lane `lane` in `[0, 4]`, this function conceptually performs the same
/// math as the scalar `Sphere::hit`, but using SIMD for the common subexpressions:
///
/// - Takes the ray origin and direction from `ray4.orig` / `ray4.dir`.
/// - Converts the sphere center (f64) and radius (f64) to f32.
/// - Solves the quadratic equation for a ray–sphere intersection.
/// - Applies the `[t_min, t_max]` interval per lane.
/// - If a valid root is found, marks `hit[lane] = true`, stores the `t` value,
///   and sets `sphere_index[lane] = sphere_idx`.
///
/// If there is no valid intersection for a lane, `hit[lane]` stays `false` and
/// `sphere_index[lane]` is left at `-1`.
pub fn hit_sphere4(
    ray4: &Ray4,
    sphere: &Sphere,
    t_min: f32,
    t_max: f32,
    sphere_idx: i32,
) -> HitInfo4 {
    // Short aliases. We copy the Vec3x4 values so we can call `dot(self, other)`
    // without borrowing issues (dot takes both parameters by value).
    let orig: Vec3x4 = ray4.orig;
    let dir:  Vec3x4 = ray4.dir;

    // Components of origin and direction as F32x4
    let ox4 = orig.x;
    let oy4 = orig.y;
    let oz4 = orig.z;

    let dx4 = dir.x;
    let dy4 = dir.y;
    let dz4 = dir.z;

    // Sphere center and radius as f32
    let center = sphere.center();
    let cx = center.x as f32;
    let cy = center.y as f32;
    let cz = center.z as f32;
    let radius = sphere.radius_f32();
    let radius2 = radius * radius;

    // Center broadcasted to all lanes
    let cx4 = F32x4::splat(cx);
    let cy4 = F32x4::splat(cy);
    let cz4 = F32x4::splat(cz);

    // oc = orig - center
    let ocx = ox4.sub(cx4);
    let ocy = oy4.sub(cy4);
    let ocz = oz4.sub(cz4);
    let oc = Vec3x4 { x: ocx, y: ocy, z: ocz };

    // a = dot(dir, dir)
    let a4 = dir.dot(dir);

    // half_b = dot(oc, dir)
    let half_b4 = oc.dot(dir);

    // c = dot(oc, oc) - radius^2
    let oc_dot = oc.dot(oc);
    let radius2_4 = F32x4::splat(radius2);
    let c4 = oc_dot.sub(radius2_4);

    // discriminant = half_b^2 - a * c
    let half_b2 = half_b4.mul(half_b4);
    let a_times_c = a4.mul(c4);
    let discriminant4 = half_b2.sub(a_times_c);

    // Convert to arrays to handle per-lane root selection and interval checks
    let disc = discriminant4.to_array();
    let a_arr = a4.to_array();
    let half_b_arr = half_b4.to_array();

    let mut info = HitInfo4::no_hit(t_max);
    let t_min_f = t_min;
    let t_max_f = t_max;

    // For each lane, emulate the scalar Sphere::hit logic as closely as possible.
    for lane in 0..4 {
        let d = disc[lane];

        if d < 0.0 {
            // No real roots -> no hit in this lane.
            continue;
        }

        let a = a_arr[lane];
        let half_b = half_b_arr[lane];

        // sqrt(discriminant)
        let sqrtd = d.sqrt();

        // Try the nearer root first
        let mut root = (-half_b - sqrtd) / a;

        if root < t_min_f || root > t_max_f {
            // Try the farther root
            root = (-half_b + sqrtd) / a;

            if root < t_min_f || root > t_max_f {
                // No valid root within [t_min, t_max]
                continue;
            }
        }

        info.hit[lane] = true;
        info.t[lane] = root;
        info.sphere_index[lane] = sphere_idx;
    }

    info
}

/// Computes the closest hit for four rays in parallel against a list of spheres.
///
/// This is the SIMD analogue of a scalar "world.hit" that iterates over all
/// spheres and keeps, for each ray, the closest intersection in the range
/// `[t_min, t_max]`.
///
/// Parameters:
/// - `ray4`: pack of 4 rays (lanes 0..3).
/// - `spheres`: slice of all spheres in the "world".
/// - `t_min`, `t_max`: parametric range to consider for valid hits.
///
/// Returns:
/// - A `HitInfo4` where, for each lane:
///     - `hit[lane]` is true if that ray hit at least one sphere,
///     - `t[lane]` is the smallest t found for that ray,
///     - `sphere_index[lane]` is the index in `spheres` of the sphere hit,
///       or -1 if there was no hit.
///
/// Implementation notes:
/// - We start with `best = HitInfo4::no_hit(t_max)` so that any real hit with
///   `t < t_max` will replace the initial value.
/// - For each sphere, we call `hit_sphere4` which returns per-lane hit info
///   for that specific sphere.
/// - Then, per lane, we update `best` if:
///     a) the new lane has a hit, and
///     b) either we had no previous hit, or this new t is smaller than the best t.
pub fn world_hit4_spheres(
    ray4: &Ray4,
    spheres: &[Sphere],
    t_min: f32,
    t_max: f32,
) -> HitInfo4 {
    // Start with "no hit" for all lanes, and t initialized to t_max.
    // This is equivalent to the scalar pattern of:
    //     let mut hit_any = false;
    //     let mut closest_so_far = t_max;
    let mut best = HitInfo4::no_hit(t_max);

    // Enumerate all spheres in the world and test them against the 4 rays.
    for (idx, sphere) in spheres.iter().enumerate() {
        let sphere_index = idx as i32;

        // SIMD intersection with this single sphere for all 4 rays.
        let info = hit_sphere4(ray4, sphere, t_min, t_max, sphere_index);

        // For each lane 0..3, decide if this sphere is a better hit than
        // what we had previously stored in `best`.
        for lane in 0..4 {
            // If this lane didn't hit this sphere, skip it.
            if !info.hit[lane] {
                continue;
            }

            let candidate_t = info.t[lane];

            // If we had no hit before on this lane, or this t is smaller
            // (i.e., closer to the ray origin), we update the best info.
            if !best.hit[lane] || candidate_t < best.t[lane] {
                best.hit[lane] = true;
                best.t[lane] = candidate_t;
                best.sphere_index[lane] = sphere_index;
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use crate::ray::Ray;
    use crate::vec3::{Point3, Vec3};
    use crate::world::hittable::{HitRecord, Hittable};
    use crate::world::material::{Lambertian, MaterialPtr};
    use crate::prelude::*;
    use crate::vec3::Color;

    fn approx_eq_f32(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn hit_sphere4_matches_scalar_hit_per_lane() {
        // 1) Simple sphere at (0, 0, -1) with radius 1
        let center = Point3::new(0.0, 0.0, -1.0);
        let radius = 1.0;
        // Dummy material just to satisfy Sphere::new
        let mat: MaterialPtr = std::rc::Rc::new(Lambertian::new(Color::new(
            0.8, 0.3, 0.3,
        )));
        let sphere = Sphere::new(center, radius, mat);

        // 2) Four different rays
        let rays = [
            // Ray straight through the center: should hit
            Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
            // Ray shifted up: likely miss
            Ray::new(Point3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
            // Ray offset in x: can be tangent or hit depending on geometry
            Ray::new(Point3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
            // Ray too far in x: should miss
            Ray::new(Point3::new(-2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
        ];

        // 3) Pack into Ray4
        let ray4 = Ray4::from_rays(rays);

        // 4) Call the SIMD version
        let t_min = 0.001_f32;
        let t_max = INFINITY as f32;
        let sphere_idx = 7; // arbitrary index to verify propagation
        let info = hit_sphere4(&ray4, &sphere, t_min, t_max, sphere_idx);

        // 5) Compare lane by lane with the scalar Sphere::hit
        let eps_t = 1e-4_f32;

        for lane in 0..4 {
            let r = &rays[lane];

            let mut rec = HitRecord::default();
            let interval = crate::interval::Interval::new(t_min as f64, t_max as f64);
            let scalar_hit = sphere.hit(r, &interval, &mut rec);

            // Bool comparison
            assert_eq!(
                scalar_hit,
                info.hit[lane],
                "Lane {}: hit mismatch: scalar={}, simd={}",
                lane,
                scalar_hit,
                info.hit[lane],
            );

            if scalar_hit {
                let t_scalar = rec.t as f32;
                let t_simd = info.t[lane];

                assert!(
                    approx_eq_f32(t_scalar, t_simd, eps_t),
                    "Lane {}: t mismatch: scalar={}, simd={}",
                    lane,
                    t_scalar,
                    t_simd,
                );

                assert_eq!(
                    sphere_idx,
                    info.sphere_index[lane],
                    "Lane {}: sphere_index mismatch: expected={}, got={}",
                    lane,
                    sphere_idx,
                    info.sphere_index[lane],
                );
            } else {
                // If scalar says "no hit", SIMD should also indicate no hit
                assert_eq!(
                    -1,
                    info.sphere_index[lane],
                    "Lane {}: expected no-hit sphere_index=-1, got={}",
                    lane,
                    info.sphere_index[lane],
                );
            }
        }
    }

    #[test]
    fn world_hit4_spheres_matches_scalar_reference() {
        use crate::interval::Interval;

        // 1) Define a small world of 3 spheres with different positions and sizes.
        let mut spheres: Vec<Sphere> = Vec::new();

        let mat1: MaterialPtr =
            std::rc::Rc::new(Lambertian::new(Color::new(0.7, 0.3, 0.3)));
        let mat2: MaterialPtr =
            std::rc::Rc::new(Lambertian::new(Color::new(0.3, 0.7, 0.3)));
        let mat3: MaterialPtr =
            std::rc::Rc::new(Lambertian::new(Color::new(0.3, 0.3, 0.7)));

        spheres.push(Sphere::new(
            Point3::new(0.0, 0.0, -1.0),
            0.5,
            mat1,
        ));
        spheres.push(Sphere::new(
            Point3::new(1.0, 0.0, -2.0),
            0.5,
            mat2,
        ));
        spheres.push(Sphere::new(
            Point3::new(-1.0, 0.0, -2.5),
            0.75,
            mat3,
        ));

        // 2) Four rays sampling different directions: some hit, some miss,
        //    and some may hit different spheres.
        let rays = [
            // Aimed roughly at the central sphere
            Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
            // Slightly to the right: more likely to intersect the sphere at x = 1
            Ray::new(Point3::new(0.2, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
            // Slightly to the left: more likely to intersect the sphere at x = -1
            Ray::new(Point3::new(-0.2, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
            // Far up: likely to miss all spheres
            Ray::new(Point3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
        ];

        let ray4 = Ray4::from_rays(rays);

        let t_min = 0.001_f32;
        let t_max = INFINITY as f32;

        // 3) SIMD world hit.
        let simd_info = world_hit4_spheres(&ray4, &spheres, t_min, t_max);

        // 4) Scalar reference: for each ray, loop over spheres and keep closest hit.
        let eps_t = 1e-4_f32;

        for lane in 0..4 {
            let r = &rays[lane];

            let mut best_t: f64 = t_max as f64;
            let mut best_idx: i32 = -1;
            let interval = Interval::new(t_min as f64, t_max as f64);
            let mut tmp_rec = HitRecord::default();

            for (idx, sphere) in spheres.iter().enumerate() {
                if sphere.hit(r, &interval, &mut tmp_rec) {
                    if tmp_rec.t < best_t {
                        best_t = tmp_rec.t;
                        best_idx = idx as i32;
                    }
                }
            }

            let simd_hit = simd_info.hit[lane];

            if best_idx == -1 {
                // Scalar saw no hit → SIMD must also report no hit.
                assert!(
                    !simd_hit,
                    "Lane {}: scalar has no hit but SIMD reports hit",
                    lane
                );
                assert_eq!(
                    -1,
                    simd_info.sphere_index[lane],
                    "Lane {}: scalar no-hit but SIMD sphere_index != -1",
                    lane
                );
            } else {
                // Scalar has a hit → SIMD must also report a hit in same lane.
                assert!(
                    simd_hit,
                    "Lane {}: scalar has hit but SIMD reports no hit",
                    lane
                );

                let t_scalar = best_t as f32;
                let t_simd = simd_info.t[lane];

                assert!(
                    approx_eq_f32(t_scalar, t_simd, eps_t),
                    "Lane {}: t mismatch: scalar={}, simd={}",
                    lane,
                    t_scalar,
                    t_simd
                );

                assert_eq!(
                    best_idx,
                    simd_info.sphere_index[lane],
                    "Lane {}: sphere index mismatch: scalar={}, simd={}",
                    lane,
                    best_idx,
                    simd_info.sphere_index[lane]
                );
            }
        }
    }

    /// Verifica que world_hit4_spheres devuelva, por lane, el mismo
    /// resultado (hay hit / no hay hit, t e índice de esfera) que
    /// recorrer las esferas de forma escalar usando Sphere::hit.
    #[test]
    fn world_hit4_spheres_matches_scalar_world_hit() {
        // ----- 1) Mundo de prueba: 3 esferas sencillas -----
        let mat: MaterialPtr = Rc::new(Lambertian::new(Color::new(0.8, 0.3, 0.3)));

        let spheres = vec![
            // Esfera central delante de la cámara
            Sphere::new(Point3::new(0.0, 0.0, -5.0), 0.5, mat.clone()),
            // Esfera desplazada a la derecha
            Sphere::new(Point3::new(2.0, 0.0, -5.0), 0.5, mat.clone()),
            // Esfera desplazada a la izquierda
            Sphere::new(Point3::new(-2.0, 0.0, -5.0), 0.5, mat.clone()),
        ];

        // ----- 2) Pack de 4 rayos escalares + Ray4 equivalente -----
        let origin = Point3::new(0.0, 0.0, 0.0);

        // Direcciones "cómodas": una al centro, una a cada esfera lateral
        // y una que falla todo (apuntando hacia arriba).
        let dir0 = Vec3::new(0.0, 0.0, -1.0); // hacia esfera 0
        let dir1 = unit_vector(Point3::new( 2.0, 0.0, -5.0) - origin); // hacia esfera 1
        let dir2 = unit_vector(Point3::new(-2.0, 0.0, -5.0) - origin); // hacia esfera 2
        let dir3 = Vec3::new(0.0, 1.0, 0.0); // apunta hacia arriba, no toca nada

        let rays_scalar = [
            Ray::new(origin, dir0),
            Ray::new(origin, dir1),
            Ray::new(origin, dir2),
            Ray::new(origin, dir3),
        ];

        // Empaquetamos en un Ray4 (SIMD)
        let ray4 = Ray4::from_rays(rays_scalar);

        let t_min = 0.001_f32;
        let t_max = 1000.0_f32;

        // ----- 3) Resultado SIMD: world_hit4_spheres -----
        let simd_info = world_hit4_spheres(&ray4, &spheres, t_min, t_max);

        // ----- 4) Referencia escalar por lane -----
        for lane in 0..4 {
            let ray = &rays_scalar[lane];

            let mut best_t = f64::INFINITY;
            let mut best_idx: i32 = -1;
            let mut tmp_rec = HitRecord::default();
            let interval = Interval::new(t_min as f64, t_max as f64);

            for (idx, sphere) in spheres.iter().enumerate() {
                if sphere.hit(ray, &interval, &mut tmp_rec) {
                    if tmp_rec.t < best_t {
                        best_t = tmp_rec.t;
                        best_idx = idx as i32;
                    }
                }
            }

            if best_idx == -1 {
                // Escalar: no hay hit en este lane.
                assert!(
                    !simd_info.hit[lane],
                    "lane {}: scalar has NO hit, but SIMD reports a hit",
                    lane
                );
                assert_eq!(
                    -1,
                    simd_info.sphere_index[lane],
                    "lane {}: scalar has NO hit, but SIMD sphere_index != -1 (got {})",
                    lane,
                    simd_info.sphere_index[lane]
                );
            } else {
                // Escalar: sí hay hit → SIMD debe reportar lo mismo.
                assert!(
                    simd_info.hit[lane],
                    "lane {}: scalar HAS a hit, but SIMD reports no hit",
                    lane
                );
                assert_eq!(
                    best_idx,
                    simd_info.sphere_index[lane],
                    "lane {}: sphere index mismatch scalar={} simd={}",
                    lane,
                    best_idx,
                    simd_info.sphere_index[lane]
                );

                let t_scalar = best_t as f32;
                let t_simd = simd_info.t[lane];
                let diff = (t_scalar - t_simd).abs();
                let eps = 1e-3_f32;

                assert!(
                    diff <= eps,
                    "lane {}: t mismatch scalar={} simd={} diff={}",
                    lane,
                    t_scalar,
                    t_simd,
                    diff
                );
            }
        }
    }

}
