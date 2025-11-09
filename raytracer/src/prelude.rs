//! `prelude` module
//!
//! Common constants, utilities, and re-exports for the ray tracer. Importing
//! this module lets you write concise code in `main.rs` and other files:
//!
//! ```rust,no_run
//! // in main.rs
//! mod prelude;
//! use crate::prelude::*;
//!
//! fn main() {
//!     let angle = 45.0;
//!     let radians = degrees_to_radians(angle);
//!     let _inf = infinity;
//!     let v = Vec3::new(1.0, 2.0, 3.0);
//!     let r = Ray::new(Point3::new(0.0, 0.0, 0.0), v);
//!     let _ = (radians, _inf, r);
//! }
//! ```

use rand::{distributions::Uniform, Rng};

// =========================
// Constants
// =========================

/// Positive infinity (`std::f64::INFINITY`).
pub const INFINITY: f64 = f64::INFINITY;

/// Archimedes' constant π
pub const PI: f64 = 3.1415926535897932385_f64;

// =======================
// Utility helper function
// =======================

/// Converts degrees to radians using `deg * π / 180`.
///
/// # Example
/// ```rust
/// # use crate::prelude::{degrees_to_radians, pi};
/// let rad = degrees_to_radians(180.0);
/// assert!((rad - pi).abs() < 1e-12);
/// ```
#[inline]
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

/// Returns a random `f64` in `[0.0, 1.0)`.
#[inline]
pub fn random_double() -> f64 {
    let mut rng = rand::thread_rng();
    // Uniform over [0.0, 1.0)
    rng.sample(Uniform::new(0.0, 1.0))
}

/// Returns a random `f64` in `[min, max)`.
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
// vec3 colors
pub use crate::vec3::Color;
// Reflect y refract
pub use crate::vec3::{reflect, refract};
// Random in unit disk
pub use crate::vec3::{random_in_unit_disk};

// rays
pub use crate::ray::Ray;

// color output helpers
pub use crate::color::write_color_to;

// hittable interface & records
pub use crate::hittable::{HitRecord, Hittable};

// concrete shapes
pub use crate::sphere::Sphere;

// containers / world lists
pub use crate::hittable_list::{HittableList, HittablePtr};

// intervals
pub use crate::interval::Interval;

// material
pub use crate::material::{Material, MaterialPtr, Lambertian, Metal, Dielectric};
