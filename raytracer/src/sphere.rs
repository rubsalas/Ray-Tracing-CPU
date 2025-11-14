//! `sphere` module
//!
//! Sphere implementation of the [`Hittable`](crate::hittable::Hittable) trait,
//! using the section 6.2 / 6.3 quadratic simplification from RTIOW.
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
use crate::material::MaterialPtr;
use crate::vec3::{dot, Point3};
use crate::hittable::{HitRecord, Hittable};

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
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, ray_t: &Interval, rec: &mut HitRecord) -> bool {
        if self.radius <= 0.0 { return false; }

        // Ray-sphere (with "h" trick)
        let oc = self.center - r.origin();
        let a = r.direction().length_squared();
        let h = dot(r.direction(), oc);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = h * h - a * c;
        if discriminant < 0.0 { return false; }
        let sqrtd = discriminant.sqrt();

        // Nearest root in the allowed interval
        let mut root = (h - sqrtd) / a;
        if !ray_t.surrounds(root) {
            root = (h + sqrtd) / a;
            if !ray_t.surrounds(root) {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);

        let outward_normal = (rec.p - self.center) / self.radius;
        rec.set_face_normal(r, outward_normal);

        // Attach material to the hit record
        rec.mat = Some(self.mat.clone());

        true
    }
}
