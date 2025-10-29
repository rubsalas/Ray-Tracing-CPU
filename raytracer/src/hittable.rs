//! `hittable` module
//!
//! Defines the intersection record [`HitRecord`] and the abstract surface
//! interface [`Hittable`]. A hittable can be intersected by a
//! [`Ray`](crate::rays::Ray) within `[t_min, t_max]`, reporting the closest hit
//! (if any) by writing into a mutable [`HitRecord`].

use crate::ray::Ray;
use crate::interval::Interval;
use crate::vec3::{dot, Point3, Vec3};

/// Intersection data produced by a successful `hit`.
///
/// Implementations fill this record when an intersection closer than any
/// previous one is found in the range `[ray_tmin, ray_tmax]`.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct HitRecord {
    /// World-space hit point `p = ray.origin + t * ray.direction`.
    pub p: Point3,
    /// Geometric surface normal at `p`. Conventionally unit length and oriented
    /// to oppose the incoming ray when [`front_face`] is `true`.
    pub normal: Vec3,
    /// Ray parameter `t` at the hit.
    pub t: f64,
    /// `true` if the intersection is on the front face (ray hits the outside
    /// of the surface), `false` if it is on the back face (ray exits).
    pub front_face: bool,
}

impl HitRecord {
    /// Sets [`front_face`](Self::front_face) and orients [`normal`](Self::normal)
    /// consistently with the incoming ray.
    ///
    /// The input `outward_normal` is assumed to be unit length and to point
    /// outward from the surface. This method flips it when the ray is inside
    /// the surface so that `normal` always opposes the ray direction on
    /// front faces.
    ///
    /// # Arguments
    /// - `r`: the incident ray
    /// - `outward_normal`: unit-length normal pointing outward
    ///
    /// # Behavior
    /// - Sets `front_face = dot(r.direction(), outward_normal) < 0`.
    /// - Sets `normal = outward_normal` if front-face, otherwise `-outward_normal`.
    #[inline]
    pub fn set_face_normal(&mut self, r: &Ray, outward_normal: Vec3) {
        self.front_face = dot(r.direction(), outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}

/// Abstract surface that can be intersected by rays.
///
/// Implementations should:
/// - Return `true` only if a hit exists in `[ray_tmin, ray_tmax]`.
/// - Populate `rec.t`, `rec.p`, and `rec.normal` (via
///   [`HitRecord::set_face_normal`]) for the closest hit in range.
/// - Leave `rec` unmodified and return `false` when no valid hit is found.
pub trait Hittable {
    /// Intersects `r` against the surface on `[ray_tmin, ray_tmax]`.
    ///
    /// On success, writes the nearest hit into `rec` and returns `true`.
    ///
    /// # Arguments
    /// - `r`: ray to test
    /// - `ray_tmin`: lower bound for `t` (use a small epsilon like `1e-8`)
    /// - `ray_tmax`: upper bound for `t` (e.g., current closest hit distance)
    /// - `rec`: output record to be filled on success
    fn hit(&self, r: &Ray, ray_t: &Interval, rec: &mut HitRecord) -> bool;
}
