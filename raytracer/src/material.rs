//! `material` module
//!
//! Materials decide how rays scatter at a hit point. The base interface exposes
//! a single method, `scatter`, which may:
//! - return `false` to indicate absorption (no scattered ray), or
//! - return `true` and fill `attenuation` and `scattered` with the next ray.

use std::rc::Rc;

use crate::ray::Ray;
use crate::prelude::*;
use crate::vec3::{Color, Vec3};
use crate::hittable::HitRecord;

/// Trait for shading and scattering behavior at surface hits.
pub trait Material {
    /// Computes the scattered ray and its attenuation.
    ///
    /// * `r_in` — incoming ray
    /// * `rec` — hit information (point, normal, `t`, face orientation)
    /// * `attenuation` — output color multiplier for this bounce
    /// * `scattered` — output ray to trace next
    /// 
    /// Returns `true` if the ray is scattered; `false` means the ray is absorbed.
    ///
    /// The default implementation absorbs all light.
    fn scatter(
        &self,
        _r_in: &Ray,
        _rec: &HitRecord,
        _attenuation: &mut Color,
        _scattered: &mut Ray,
    ) -> bool
    {
        false
    }
}

/// Shared-pointer alias for materials (single-threaded).
pub type MaterialPtr = Rc<dyn Material>;

/// Lambertian (diffuse) material — Listing 61
///
/// Scatters rays in a random direction over the hemisphere centered on the
/// surface normal. The attenuation equals the material's `albedo`.
#[derive(Clone, Debug)]
pub struct Lambertian {
    albedo: Color,
}

impl Lambertian {
    /// Creates a Lambertian material with the given diffuse `albedo` color.
    #[inline]
    pub fn new(albedo: Color) -> Self {
        Self { albedo }
    }
}

impl Material for Lambertian {
    /// Scatter by choosing a random hemisphere direction about the hit normal.
    ///
    /// - `scattered.origin = rec.p`
    /// - `scattered.direction = rec.normal + random_unit_vector()`
    /// - `attenuation = albedo`
    ///
    /// Always returns `true` (purely diffuse).
    fn scatter(
        &self,
        _r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        let mut scatter_direction = rec.normal + Vec3::random_unit_vector();

        // Catch degenerate scatter direction
        if scatter_direction.near_zero() {
            scatter_direction = rec.normal;
        }

        *scattered = Ray::new(rec.p, scatter_direction);
        *attenuation = self.albedo;
        true
    }
}

/// Rough (fuzzy) metal
///
/// Perfect specular reflection when `fuzz == 0`.
/// As `fuzz` increases toward 1, the reflection direction is randomized by
/// adding `fuzz * random_unit_vector()` (a unit vector noise).
#[derive(Clone, Debug)]
pub struct Metal {
    albedo: Color,
    fuzz: f64, // clamped to [0, 1]
}

impl Metal {
    /// Creates a metal with `albedo` and surface roughness `fuzz` in [0, 1].
    #[inline]
    pub fn new(albedo: Color, fuzz: f64) -> Self {
        Self {
            albedo,
            fuzz: fuzz.clamp(0.0, 1.0), // mimic fuzz < 1 ? fuzz : 1
        }
    }
}

impl Material for Metal {
    /// Reflect about the normal, then add fuzzy noise, and accept only if
    /// the scattered ray still goes outward (`dot(dir, normal) > 0`).
    fn scatter(
        &self,
        r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        let mut reflected = Vec3::reflect(r_in.direction(), rec.normal);
        // Listing 69: normalize reflected, then add fuzz * random_unit_vector()
        reflected = unit_vector(reflected) + self.fuzz * Vec3::random_unit_vector();

        *scattered = Ray::new(rec.p, reflected);
        *attenuation = self.albedo;

        dot(scattered.direction(), rec.normal) > 0.0
    }
}
