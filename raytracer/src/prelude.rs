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

// ============================
// Re-exports (common headers)
// ============================

// vec3 / math
pub use crate::vec3::{cross, dot, unit_vector, Point3, Vec3};
// vec3 colors
pub use crate::vec3::Color;

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
