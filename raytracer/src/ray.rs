//! `ray` module
//!
//! A geometric ray is a parametric half-line defined as
//! `P(t) = origin + t * direction`, where `t ≥ 0` typically denotes points
//! visible “in front of” the origin.
//!
//! # Notes
//! - The `direction` does not have to be unit length. If it is normalized,
//!   then `t` numerically matches distance; otherwise, `t` is distance scaled
//!   by `|direction|`.
//! - Secondary rays should usually use a small lower bound (e.g. `t_min=1e-8`)
//!   to avoid self-intersections due to floating-point error.

use crate::vec3::{Point3, Vec3};

/// Parametric ray with an origin and a direction.
///
/// The direction may or may not be normalized depending on the use site.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ray {
    /// Ray origin (3D point).
    orig: Point3,
    /// Ray direction (3D vector). Can be normalized or not.
    dir: Vec3,
}

impl Ray {
    /// Creates a new ray from `origin` and `direction`.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use crate::vec3::{Point3, Vec3};
    /// # use crate::ray::Ray;
    /// let r = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
    /// ```
    #[inline]
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Self { orig: origin, dir: direction }
    }

    /// Returns the ray origin (by value; `Point3` is `Copy`).
    #[inline]
    pub fn origin(&self) -> Point3 {
        self.orig
    }

    /// Returns the ray direction (by value; `Vec3` is `Copy`).
    #[inline]
    pub fn direction(&self) -> Vec3 {
        self.dir
    }

    /// Evaluates the point along the ray at parameter `t`:
    /// `origin + t * direction`.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use crate::vec3::{Point3, Vec3};
    /// # use crate::ray::Ray;
    /// let r = Ray::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 1.0));
    /// assert_eq!(r.at(4.0), Point3::new(1.0, 2.0, 7.0));
    /// ```
    #[inline]
    pub fn at(&self, t: f64) -> Point3 {
        self.orig + t * self.dir
    }
}
