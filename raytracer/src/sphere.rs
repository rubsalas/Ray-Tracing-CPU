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
//! - The constructor clamps negative radii to zero (matching the C++ `fmax(0, r)`).
//! - Normal is computed as `(p - center) / radius`, which is unit length for
//!   `radius > 0`. For `radius == 0` the sphere degenerates and will not report hits.

use crate::hittable::{HitRecord, Hittable};
use crate::rays::Ray;
use crate::vec3::{dot, Point3, Vec3};

/// Solid sphere defined by a `center` and a non-negative `radius`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sphere {
    center: Point3,
    radius: f64,
}

impl Sphere {
    /// Creates a new sphere; negative radii are clamped to zero.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use crate::vec3::Point3;
    /// # use crate::sphere::Sphere;
    /// let s = Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5);
    /// ```
    pub fn new(center: Point3, radius: f64) -> Self {
        Self {
            center,
            radius: radius.max(0.0),
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
    /// Ray–sphere intersection using the 6.2/6.3 simplification:
    ///
    /// Let `oc = center - r.origin()`, `a = |D|²`, `h = D·oc`, `c = |oc|² - r²`.
    /// The discriminant is `Δ = h² - a·c`. For `Δ < 0` there is no hit.
    /// Otherwise, candidate roots are `(h ± √Δ) / a`.
    ///
    /// The nearest root inside `[ray_tmin, ray_tmax]` is chosen; on success,
    /// `rec.t`, `rec.p`, and `rec.normal` are filled and the function returns `true`.
    fn hit(&self, r: &Ray, ray_tmin: f64, ray_tmax: f64, rec: &mut HitRecord) -> bool {
        // Degenerate sphere (radius == 0) never hits meaningfully.
        if self.radius <= 0.0 {
            return false;
        }

        // oc = center - origin  (matches the book’s 6.2 form)
        let oc: Vec3 = self.center - r.origin();

        let a = r.direction().length_squared();                 // a = |D|^2
        let h = dot(r.direction(), oc);                         // h = D · oc
        let c = oc.length_squared() - self.radius * self.radius; // c = |oc|^2 - r^2

        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return false;
        }

        let sqrtd = discriminant.sqrt();

        // Find the nearest root within the allowed range.
        let mut root = (h - sqrtd) / a;
        if root <= ray_tmin || ray_tmax <= root {
            root = (h + sqrtd) / a;
            if root <= ray_tmin || ray_tmax <= root {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        
        // Compute outward_normal and orient via set_face_normal
        let outward_normal = (rec.p - self.center) / self.radius;
        rec.set_face_normal(r, outward_normal);

        true
    }
}
