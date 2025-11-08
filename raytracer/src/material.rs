//! `material` module
//!
//! Materials decide how rays scatter at a hit point. The base interface exposes
//! a single method, `scatter`, which may:
//! - return `false` to indicate absorption (no scattered ray), or
//! - return `true` and fill `attenuation` and `scattered` with the next ray.

use std::rc::Rc;

use crate::ray::Ray;
use crate::prelude::*;
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
        let mut reflected = reflect(r_in.direction(), rec.normal);
        // Listing 69: normalize reflected, then add fuzz * random_unit_vector()
        reflected = unit_vector(reflected) + self.fuzz * Vec3::random_unit_vector();

        *scattered = Ray::new(rec.p, reflected);
        *attenuation = self.albedo;

        dot(scattered.direction(), rec.normal) > 0.0
    }
}

/// Dielectric (full glass)
///
/// Uses Snell refraction, total internal reflection, and Schlick's
/// approximation to choose between reflection and refraction.
#[derive(Clone, Debug)]
pub struct Dielectric {
    /// Refractive index (η)
    refraction_index: f64,
}

impl Dielectric {
    #[inline]
    pub fn new(eta: f64) -> Self {
        Self { refraction_index: eta }
    }

    /// Schlick's approximation for reflectance.
    #[inline]
    fn reflectance(cosine: f64, refraction_index: f64) -> f64 {
        // r0 = ((1 - n) / (1 + n))^2
        let mut r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
        r0 *= r0;
        // R(θ) ≈ r0 + (1 - r0)(1 - cosθ)^5
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(
        &self,
        r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        *attenuation = Color::new(1.0, 1.0, 1.0); // no absorption in this simple model

        // Relative IOR (air→material or material→air)
        let ri = if rec.front_face { 1.0 / self.refraction_index } else { self.refraction_index };

        let unit_direction = unit_vector(r_in.direction());
        let cos_theta = f64::min(dot(-unit_direction, rec.normal), 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        // Total internal reflection?
        let cannot_refract = ri * sin_theta > 1.0;

        // Schlick: probabilistic reflect vs refract
        let direction = if cannot_refract || Self::reflectance(cos_theta, ri) > random_double() {
            reflect(unit_direction, rec.normal)
        } else {
            refract(unit_direction, rec.normal, ri)
        };

        *scattered = Ray::new(rec.p, direction);
        true
    }
}
