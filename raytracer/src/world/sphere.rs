//! `sphere` module
//!
//! Sphere implementation of the [`Hittable`](crate::hittable::Hittable) trait,
//!
//! The implicit surface equation is ‖P − C‖² = r². Intersecting the ray
//! `P(t) = O + t·D` yields a quadratic in `t`. Using the *h = D·oc* form
//! reduces precision issues and avoids carrying a factor 2.
//!
//! # Notes
//! - The constructor clamps negative radii to zero.
//! - Normal is computed as `(p - center) / radius`, which is unit length for
//!   `radius > 0`. For `radius == 0` the sphere degenerates and will not report hits.

use crate::ray::Ray;
use crate::interval::Interval;
use crate::vec3::{dot, Point3};
use crate::world::material::MaterialPtr;
use crate::metrics::core_stats::with_core_stats;
use crate::world::hittable::{HitRecord, Hittable};

/// Solid sphere defined by a `center` and a non-negative `radius`.
#[derive(Clone)]
pub struct Sphere {
    center: Point3,
    radius: f64,
    mat: MaterialPtr,
}

impl Sphere {
    /// Creates a sphere with center, non-negative radius, and a material.
    pub fn new(center: Point3, radius: f64, mat: MaterialPtr) -> Self {
        Self {
            center,
            radius: radius.max(0.0),
            mat,
        }
    }

    /// Returns the sphere center.
    #[inline]
    pub fn center(&self) -> Point3 {
        self.center
    }

    /// Returns the (non-negative) sphere radius.
    #[inline]
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns the radius of the sphere as f32, suitable for SIMD math.
    #[inline]
    pub fn radius_f32(&self) -> f32 {
        self.radius as f32
    }
}

impl Hittable for Sphere {
    fn hit(
        &self,
        r: &Ray,
        ray_t: &Interval,
        rec: &mut HitRecord,
    ) -> bool {
        // Se registra un test escalar de intersección.
        with_core_stats(|stats| {
            stats.scalar_intersection_tests += 1;
        });

        let hit_occurred = {
            let oc = r.origin() - self.center;
            let a = r.direction().length_squared();
            let half_b = dot(oc, r.direction());
            let c = oc.length_squared() - self.radius * self.radius;

            let discriminant = half_b * half_b - a * c;
            if discriminant < 0.0 {
                false
            } else {
                let sqrtd = discriminant.sqrt();

                // Se intenta t1 y t2 como en el libro.
                let mut root = (-half_b - sqrtd) / a;
                if !ray_t.surrounds(root) {
                    root = (-half_b + sqrtd) / a;
                    if !ray_t.surrounds(root) {
                        false
                    } else {
                        // se rellena el HitRecord para root
                        rec.t = root;
                        rec.p = r.at(rec.t);
                        let outward_normal = (rec.p - self.center) / self.radius;
                        rec.set_face_normal(r, outward_normal);
                        rec.mat = Some(self.mat.clone());
                        true
                    }
                } else {
                    rec.t = root;
                    rec.p = r.at(rec.t);
                    let outward_normal = (rec.p - self.center) / self.radius;
                    rec.set_face_normal(r, outward_normal);
                    rec.mat = Some(self.mat.clone());
                    true
                }
            }
        };

        // Se actualiza el contador de hit o miss.
        with_core_stats(|stats| {
            if hit_occurred {
                stats.scalar_intersection_hits += 1;
            } else {
                stats.scalar_intersection_misses += 1;
            }
        });

        hit_occurred
    }
}
