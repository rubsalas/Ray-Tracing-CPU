//! `interval` module
//!
//! Closed or half-open numeric ranges used throughout the ray tracer for
//! parameter constraints (e.g., valid `t` intervals on ray hits).
//!
//! This mirrors the RTIOW `interval` class: it stores `min` and `max`, can
//! report size, and provides inclusive/exclusive membership tests.
//!
//! # Examples
//! ```rust
//! # use crate::interval::Interval;
//! let iv = Interval::new(0.0, 1.0);
//! assert!(iv.contains(0.0));   // inclusive
//! assert!(iv.contains(1.0));   // inclusive
//! assert!(iv.surrounds(0.5));  // strict interior
//! assert!(!iv.surrounds(0.0)); // endpoints excluded
//! assert_eq!(iv.size(), 1.0);
//!
//! // Predefined intervals
//! let e = Interval::EMPTY;     // empty
//! let u = Interval::UNIVERSE;  // (-∞, +∞)
//! assert!(e.size().is_sign_negative() || e.size() == 0.0);
//! assert!(u.contains(f64::INFINITY) == false); // ∞ is not ≤ ∞ numerically
//! ```

/// Numeric interval with `min` and `max` bounds.
///
/// By convention, the default interval is empty, matching the RTIOW
/// constructor that sets `min = +∞` and `max = -∞`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Interval {
    /// Lower bound.
    pub min: f64,
    /// Upper bound.
    pub max: f64,
}

impl Interval {
    /// Creates an empty interval: `(min=+∞, max=-∞)`.
    ///
    /// This matches the book’s default constructor.
    #[inline]
    pub fn empty() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Creates an interval with explicit `min` and `max`.
    ///
    /// This does not reorder the inputs; if `min > max`, the interval is
    /// considered invalid/empty by convention.
    #[inline]
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    /// Interval length: `max - min`.
    #[inline]
    pub fn size(&self) -> f64 {
        self.max - self.min
    }

    /// Inclusive membership test: `min ≤ x ≤ max`.
    #[inline]
    pub fn contains(&self, x: f64) -> bool {
        self.min <= x && x <= self.max
    }

    /// Strict interior membership: `min < x < max`.
    #[inline]
    pub fn surrounds(&self, x: f64) -> bool {
        self.min < x && x < self.max
    }

    /// Clamps `x` to this interval: returns `min` if `x < min`,
    /// `max` if `x > max`, otherwise returns `x`.
    #[inline]
    pub fn clamp(&self, x: f64) -> f64 {
        if x < self.min {
            self.min
        } else if x > self.max {
            self.max
        } else {
            x
        }
    }

    /// Predefined empty interval: `(min=+∞, max=-∞)`.
    pub const EMPTY: Self = Self {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
    };

    /// Predefined universe interval: `(-∞, +∞)`.
    pub const UNIVERSE: Self = Self {
        min: f64::NEG_INFINITY,
        max: f64::INFINITY,
    };
}

impl Default for Interval {
    /// `Default` returns an empty interval, matching RTIOW.
    fn default() -> Self {
        Self::empty()
    }
}
