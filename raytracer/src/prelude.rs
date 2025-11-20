use rand::{distributions::Uniform, Rng};

// =========================
// Constants
// =========================

pub const INFINITY: f64 = f64::INFINITY;
pub const PI: f64 = 3.1415926535897932385_f64;

// =======================
// Utility helper function
// =======================

#[inline]
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

#[inline]
pub fn random_double() -> f64 {
    let mut rng = rand::thread_rng();
    rng.sample(Uniform::new(0.0, 1.0))
}

#[inline]
pub fn random_double_range(min: f64, max: f64) -> f64 {
    let mut rng = rand::thread_rng();
    rng.sample(Uniform::new(min, max))
}

// ============================
// Re-exports (common headers)
// ============================

// vec3 / math
pub use crate::vec3::{cross, dot, unit_vector, Point3, Vec3};
pub use crate::vec3::Color;
pub use crate::vec3::{reflect, refract};
pub use crate::vec3::random_in_unit_disk;

// rays
pub use crate::ray::Ray;

// color output helpers
pub use crate::image::color::write_color_to;
// (también podrías usar crate::image::write_color_to si re-exportas en image::mod)

// hittable interface & records (ahora en world/)
pub use crate::world::hittable::{HitRecord, Hittable};

// concrete shapes
pub use crate::world::sphere::Sphere;

// containers / world lists
pub use crate::world::hittable_list::{HittableList, HittablePtr};

// intervals (sigue en la raíz)
pub use crate::interval::Interval;

// material (ahora en world/)
pub use crate::world::material::{Material, MaterialPtr, Lambertian, Metal, Dielectric};
