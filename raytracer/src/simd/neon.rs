// src/simd/neon.rs

//! NEON-based SIMD primitives for `aarch64` targets.
//!
//! This module provides small, focused SIMD types that wrap NEON intrinsics
//! from `core::arch::aarch64` and are intended to be used as building blocks
//! for vectorized parts of the ray tracer.
//!
//! The main design goals are:
//! - Keep the public interface simple and focused on math operations.
//! - Hide the raw NEON intrinsics behind safe(ish) wrappers.
//! - Provide unit tests that compare the SIMD results against scalar
//!   reference implementations.

// This entire module is only compiled on 64-bit ARM (aarch64) targets.
// For other architectures (e.g., x86_64) this file is ignored.
#![cfg(target_arch = "aarch64")]

// While the SIMD types are not yet used in the main render path, we still
// want them compiled and tested, so we silence the "dead_code" warnings.
#![allow(dead_code)]

use core::arch::aarch64::*;

/// A wrapper around a NEON vector of four `f32` values (`float32x4_t`).
///
/// This type represents four single-precision floating-point numbers packed
/// into a single SIMD register. All operations are lane-wise, meaning that
/// each operation is applied independently to the four lanes.
///
/// Typical usage in this project:
/// - Represent four independent scalar values that can be processed in
///   parallel with NEON intrinsics.
/// - Serve as the scalar building block for higher-level SIMD types such
///   as [`Vec3x4`], which packs four 3D vectors.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct F32x4(pub float32x4_t);

impl F32x4 {
    /// Creates a vector where all four lanes are set to the same `value`.
    ///
    /// This is equivalent to:
    /// ```text
    /// [value, value, value, value]
    /// ```
    ///
    /// Internally uses the NEON intrinsic `vdupq_n_f32`.
    #[inline]
    pub fn splat(value: f32) -> Self {
        unsafe { F32x4(vdupq_n_f32(value)) }
    }

    /// Creates an [`F32x4`] from an array of four `f32` values.
    ///
    /// The array is loaded into a NEON register using `vld1q_f32`.
    /// The ordering of the lanes is preserved:
    /// `values[0]` becomes lane 0, `values[1]` lane 1, and so on.
    #[inline]
    pub fn from_array(values: [f32; 4]) -> Self {
        unsafe { F32x4(vld1q_f32(values.as_ptr())) }
    }

    /// Stores the four lanes into a `[f32; 4]` array.
    ///
    /// This is the inverse operation of [`Self::from_array`]. It is primarily
    /// used in tests and debugging to compare SIMD results against scalar
    /// reference implementations.
    #[inline]
    pub fn to_array(self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        unsafe {
            vst1q_f32(out.as_mut_ptr(), self.0);
        }
        out
    }

    /// Lane-wise addition: `self + other`.
    ///
    /// Each lane `i` of the result equals `self[i] + other[i]`.
    /// Internally uses the NEON intrinsic `vaddq_f32`.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        unsafe { F32x4(vaddq_f32(self.0, other.0)) }
    }

    /// Lane-wise subtraction: `self - other`.
    ///
    /// Each lane `i` of the result equals `self[i] - other[i]`.
    /// Internally uses the NEON intrinsic `vsubq_f32`.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        unsafe { F32x4(vsubq_f32(self.0, other.0)) }
    }

    /// Lane-wise multiplication: `self * other`.
    ///
    /// Each lane `i` of the result equals `self[i] * other[i]`.
    /// Internally uses the NEON intrinsic `vmulq_f32`.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        unsafe { F32x4(vmulq_f32(self.0, other.0)) }
    }

    /// Lane-wise division: `self / other`.
    ///
    /// Each lane `i` of the result equals `self[i] / other[i]`.
    /// Internally uses the NEON intrinsic `vdivq_f32`. The exact
    /// implementation may use reciprocal approximations followed by
    /// refinement, but for typical ray-tracing computations this
    /// provides sufficient accuracy.
    #[inline]
    pub fn div(self, other: Self) -> Self {
        unsafe { F32x4(vdivq_f32(self.0, other.0)) }
    }

    /// Lane-wise square root: `sqrt(self)`.
    ///
    /// Each lane `i` of the result equals `self[i].sqrt()`.
    /// Internally uses the NEON intrinsic `vsqrtq_f32`.
    #[inline]
    pub fn sqrt(self) -> Self {
        unsafe { F32x4(vsqrtq_f32(self.0)) }
    }
}

// -----------------------------------------------------------------------------
// Operator overloads for F32x4
// -----------------------------------------------------------------------------

impl core::ops::Add for F32x4 {
    type Output = Self;

    /// Shorthand for lane-wise addition using the `+` operator.
    ///
    /// Equivalent to [`F32x4::add`].
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl core::ops::Sub for F32x4 {
    type Output = Self;

    /// Shorthand for lane-wise subtraction using the `-` operator.
    ///
    /// Equivalent to [`F32x4::sub`].
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}

impl core::ops::Mul for F32x4 {
    type Output = Self;

    /// Shorthand for lane-wise multiplication using the `*` operator.
    ///
    /// Equivalent to [`F32x4::mul`].
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

impl core::ops::Div for F32x4 {
    type Output = Self;

    /// Shorthand for lane-wise division using the `/` operator.
    ///
    /// Equivalent to [`F32x4::div`].
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.div(rhs)
    }
}

/// A SIMD structure holding four 3D vectors in a Structure of Arrays (SoA) layout.
///
/// Conceptually, this type represents four separate vectors:
///
/// ```text
/// v0 = (x0, y0, z0)
/// v1 = (x1, y1, z1)
/// v2 = (x2, y2, z2)
/// v3 = (x3, y3, z3)
/// ```
///
/// but they are stored as three [`F32x4`] registers:
///
/// - `x = [x0, x1, x2, x3]`
/// - `y = [y0, y1, y2, y3]`
/// - `z = [z0, z1, z2, z3]`
///
/// This layout is convenient for SIMD because operations on all four vectors
/// can be expressed as lane-wise operations over `x`, `y`, and `z` separately.
#[derive(Copy, Clone)]
pub struct Vec3x4 {
    /// SIMD vector containing the X components of the four 3D vectors.
    pub x: F32x4,
    /// SIMD vector containing the Y components of the four 3D vectors.
    pub y: F32x4,
    /// SIMD vector containing the Z components of the four 3D vectors.
    pub z: F32x4,
}

impl Vec3x4 {
    /// Constructs a new [`Vec3x4`] from three SIMD registers.
    ///
    /// This is the most direct constructor when you already have
    /// three [`F32x4`] values representing the X, Y, and Z components.
    #[inline]
    pub fn new(x: F32x4, y: F32x4, z: F32x4) -> Self {
        Self { x, y, z }
    }

    /// Constructs a [`Vec3x4`] from three arrays of four `f32` each.
    ///
    /// Each array represents one coordinate component (X, Y, or Z) for all four
    /// vectors. The data is interpreted as:
    ///
    /// - `xs = [x0, x1, x2, x3]`
    /// - `ys = [y0, y1, y2, y3]`
    /// - `zs = [z0, z1, z2, z3]`
    ///
    /// and internally converted to a SoA layout using [`F32x4::from_array`].
    #[inline]
    pub fn from_arrays(xs: [f32; 4], ys: [f32; 4], zs: [f32; 4]) -> Self {
        Self {
            x: F32x4::from_array(xs),
            y: F32x4::from_array(ys),
            z: F32x4::from_array(zs),
        }
    }

    /// Component-wise addition of two `Vec3x4` values.
    ///
    /// For each lane `i`, this computes:
    ///
    /// ```text
    /// result_i = self_i + other_i
    ///          = (x_i + x'_i, y_i + y'_i, z_i + z'_i)
    /// ```
    ///
    /// where `self_i` and `other_i` are the i-th 3D vectors in each packet.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Component-wise subtraction of two `Vec3x4` values.
    ///
    /// For each lane `i`, this computes:
    ///
    /// ```text
    /// result_i = self_i - other_i
    ///          = (x_i - x'_i, y_i - y'_i, z_i - z'_i)
    /// ```
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Multiplies each lane of this vector by a scalar value stored
    /// in an [`F32x4`], lane-wise.
    ///
    /// If `s` is a vector where all lanes hold the same scalar, this
    /// effectively scales each of the four 3D vectors by the same amount.
    /// If `s` has different values in each lane, each vector gets its own
    /// per-lane scale factor.
    #[inline]
    pub fn mul_scalar(self, s: F32x4) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Lane-wise dot product of two `Vec3x4` values.
    ///
    /// For each lane `i`, this computes:
    ///
    /// ```text
    /// dot_i = self_i · other_i
    ///       = x_i * x'_i + y_i * y'_i + z_i * z'_i
    /// ```
    ///
    /// The result is returned as an [`F32x4`] where each lane holds the dot
    /// product of one pair of 3D vectors.
    #[inline]
    pub fn dot(self, other: Self) -> F32x4 {
        let xx = self.x * other.x;
        let yy = self.y * other.y;
        let zz = self.z * other.z;
        xx + yy + zz
    }

    /// Lane-wise length (magnitude) of each 3D vector.
    ///
    /// Internally this computes:
    ///
    /// ```text
    /// length_i = sqrt(self_i · self_i)
    /// ```
    ///
    /// The result is returned as an [`F32x4`] where each lane holds the length
    /// of the corresponding vector.
    #[inline]
    pub fn length(self) -> F32x4 {
        let len2 = self.dot(self);
        len2.sqrt()
    }

    /// Normalizes each of the four 3D vectors to unit length (as far as
    /// floating-point arithmetic allows).
    ///
    /// For each lane `i`, this computes:
    ///
    /// ```text
    /// unit_i = self_i / |self_i|
    /// ```
    ///
    /// If any lane has zero length, the behavior is similar to scalar
    /// division by zero (it may produce `NaN` or `inf`), just like the
    /// scalar version would. It is the caller's responsibility to ensure
    /// non-zero length if needed.
    #[inline]
    pub fn unit_vector(self) -> Self {
        let len = self.length();
        let ones = F32x4::splat(1.0);
        let inv_len = ones / len;
        self.mul_scalar(inv_len)
    }
}

// -----------------------------------------------------------------------------
// Unit tests: SIMD vs scalar reference implementations
// -----------------------------------------------------------------------------
//
// These tests are compiled and executed only when running `cargo test` on an
// `aarch64` target. They validate that the NEON-backed operations behave as
// expected when compared to straightforward scalar computations.

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple helper for approximate floating-point comparisons.
    ///
    /// Returns `true` if the absolute difference between `a` and `b`
    /// is less than or equal to `eps`.
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Verifies that `F32x4` lane-wise addition matches scalar addition
    /// for four independent values.
    #[test]
    fn f32x4_add_matches_scalar() {
        // Scalar inputs.
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];

        // Scalar reference result.
        let expected = [
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
        ];

        // SIMD computation.
        let va = F32x4::from_array(a);
        let vb = F32x4::from_array(b);
        let vc = va + vb;
        let got = vc.to_array();

        // Compare each lane with a small tolerance.
        for i in 0..4 {
            assert!(
                approx_eq(expected[i], got[i], 1e-6),
                "lane {}: expected {}, got {}",
                i,
                expected[i],
                got[i]
            );
        }
    }

    /// Verifies that `Vec3x4::dot` matches scalar dot products for
    /// four pairs of 3D vectors.
    #[test]
    fn vec3x4_dot_matches_scalar() {
        // Four scalar 3D vectors v_i.
        let v = [
            [1.0f32, 2.0, 3.0],
            [-1.0, 0.5, 4.0],
            [0.0, 1.0, 0.0],
            [2.0, -1.0, 0.5],
        ];

        // Four scalar 3D vectors w_i.
        let w = [
            [0.5f32, -1.0, 2.0],
            [1.0, 1.0, 1.0],
            [3.0, 0.0, -1.0],
            [-2.0, 4.0, 0.0],
        ];

        // Scalar reference dot products.
        let mut expected = [0.0f32; 4];
        for i in 0..4 {
            expected[i] = v[i][0] * w[i][0]
                        + v[i][1] * w[i][1]
                        + v[i][2] * w[i][2];
        }

        // Build SoA arrays for Vec3x4.
        let vx = [v[0][0], v[1][0], v[2][0], v[3][0]];
        let vy = [v[0][1], v[1][1], v[2][1], v[3][1]];
        let vz = [v[0][2], v[1][2], v[2][2], v[3][2]];

        let wx = [w[0][0], w[1][0], w[2][0], w[3][0]];
        let wy = [w[0][1], w[1][1], w[2][1], w[3][1]];
        let wz = [w[0][2], w[1][2], w[2][2], w[3][2]];

        // SIMD vectors.
        let vv = Vec3x4::from_arrays(vx, vy, vz);
        let ww = Vec3x4::from_arrays(wx, wy, wz);

        // SIMD dot products (one per lane).
        let dots = vv.dot(ww).to_array();

        // Compare each lane with a tolerance.
        for i in 0..4 {
            assert!(
                approx_eq(expected[i], dots[i], 1e-5),
                "lane {}: expected {}, got {}",
                i,
                expected[i],
                dots[i]
            );
        }
    }

    /// Verifies that `Vec3x4::unit_vector` matches scalar normalization
    /// for four non-zero 3D vectors.
    #[test]
    fn vec3x4_unit_vector_matches_scalar() {
        // Four scalar 3D vectors (none of them is zero-length).
        let v = [
            [1.0f32, 0.0, 0.0],
            [0.0, 3.0, 4.0],
            [1.0, 2.0, 2.0],
            [-2.0, 0.0, 2.0],
        ];

        // Scalar reference: normalized vectors.
        let mut expected = [[0.0f32; 3]; 4];
        for i in 0..4 {
            let len = (v[i][0] * v[i][0]
                     + v[i][1] * v[i][1]
                     + v[i][2] * v[i][2]).sqrt();
            expected[i][0] = v[i][0] / len;
            expected[i][1] = v[i][1] / len;
            expected[i][2] = v[i][2] / len;
        }

        // Build SoA arrays for Vec3x4.
        let vx = [v[0][0], v[1][0], v[2][0], v[3][0]];
        let vy = [v[0][1], v[1][1], v[2][1], v[3][1]];
        let vz = [v[0][2], v[1][2], v[2][2], v[3][2]];

        let vv = Vec3x4::from_arrays(vx, vy, vz);
        let uu = vv.unit_vector();

        // Extract SIMD results back to scalar arrays.
        let ux = uu.x.to_array();
        let uy = uu.y.to_array();
        let uz = uu.z.to_array();

        // Compare each component of each lane with a tolerance.
        for i in 0..4 {
            assert!(
                approx_eq(expected[i][0], ux[i], 1e-4),
                "lane {} x: expected {}, got {}",
                i,
                expected[i][0],
                ux[i]
            );
            assert!(
                approx_eq(expected[i][1], uy[i], 1e-4),
                "lane {} y: expected {}, got {}",
                i,
                expected[i][1],
                uy[i]
            );
            assert!(
                approx_eq(expected[i][2], uz[i], 1e-4),
                "lane {} z: expected {}, got {}",
                i,
                expected[i][2],
                uz[i]
            );
        }
    }
}
