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

/// Perfect specular metal — Listing 65
///
/// Reflects the incoming direction about the surface normal and continues
/// tracing with that reflected ray. Attenuation equals the material `albedo`.
#[derive(Clone, Debug)]
pub struct Metal {
    albedo: Color,
}

impl Metal {
    /// Creates a perfect-mirror metal with the given `albedo`.
    #[inline]
    pub fn new(albedo: Color) -> Self {
        Self { albedo }
    }
}

impl Material for Metal {
    /// Scatter as a perfect reflection about the hit normal.
    ///
    /// - `scattered.origin = rec.p`
    /// - `scattered.direction = reflect(r_in.direction(), rec.normal)`
    /// - `attenuation = albedo`
    fn scatter(
        &self,
        r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        let reflected = Vec3::reflect(r_in.direction(), rec.normal);
        *scattered = Ray::new(rec.p, reflected);
        *attenuation = self.albedo;
        true
    }
}
