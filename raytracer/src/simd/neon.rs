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

use crate::ray::Ray;
use crate::camera::Camera;
use crate::vec3::{Vec3, Point3};


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

    /// Lane-wise minimum: for each lane `i`, result[i] = min(self[i], other[i]).
    ///
    /// This is used as a building block for SIMD clamp operations, where the
    /// scalar version uses `value.clamp(min, max)` and we want the same effect
    /// on four lanes at once.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        unsafe { F32x4(vminq_f32(self.0, other.0)) }
    }

    /// Lane-wise maximum: for each lane `i`, result[i] = max(self[i], other[i]).
    ///
    /// Together with [`F32x4::min`], this allows us to implement:
    ///   clamp(x, min, max) = x.max(min).min(max)
    /// in a SIMD-friendly way.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        unsafe { F32x4(vmaxq_f32(self.0, other.0)) }
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

/// A bundle of 4 rays stored in SIMD-friendly form.
///
/// Each field is a Vec3x4, which internally holds three F32x4 vectors:
/// - `orig`: 4 origins (one per lane),
/// - `dir`:  4 directions (one per lane).
///
/// In this first phase (4.1), Ray4 is mainly used as a container:
/// we still build rays using the scalar Camera::get_ray, then pack them
/// into Ray4 and later unpack them back to 4 scalar `Ray` values.
#[derive(Clone, Copy)]
pub struct Ray4 {
    pub orig: Vec3x4,
    pub dir:  Vec3x4,
}

impl Ray4 {
    /// Packs four scalar rays into a Ray4 by converting the f64 components
    /// to f32 and storing them into Vec3x4 / F32x4 lanes.
    ///
    /// This is a transitional helper: the goal in later stages is to
    /// build Ray4 directly from SIMD math in the camera, but for now we
    /// reuse the existing scalar `Ray` construction.
    pub fn from_rays(rays: [Ray; 4]) -> Self {
        // Arrays of 4 lanes for origin and direction, in f32.
        let mut ox = [0.0f32; 4];
        let mut oy = [0.0f32; 4];
        let mut oz = [0.0f32; 4];

        let mut dx = [0.0f32; 4];
        let mut dy = [0.0f32; 4];
        let mut dz = [0.0f32; 4];

        for lane in 0..4 {
            let r = &rays[lane];

            // We rely on the Ray API (origin() and direction()).
            let o = r.origin();
            let d = r.direction();

            ox[lane] = o.x as f32;
            oy[lane] = o.y as f32;
            oz[lane] = o.z as f32;

            dx[lane] = d.x as f32;
            dy[lane] = d.y as f32;
            dz[lane] = d.z as f32;
        }

        Ray4 {
            orig: Vec3x4 {
                x: F32x4::from_array(ox),
                y: F32x4::from_array(oy),
                z: F32x4::from_array(oz),
            },
            dir: Vec3x4 {
                x: F32x4::from_array(dx),
                y: F32x4::from_array(dy),
                z: F32x4::from_array(dz),
            },
        }
    }

    /// Extracts a single scalar `Ray` from lane `lane` (0..3).
    ///
    /// This converts the internal f32 lane values back to f64 and
    /// rebuilds a standard `Ray` using `Ray::new`.
    pub fn lane(&self, lane: usize) -> Ray {
        debug_assert!(lane < 4, "Ray4::lane index out of bounds");

        // Convert SIMD vectors back to arrays so we can pick a single lane.
        let ox = self.orig.x.to_array();
        let oy = self.orig.y.to_array();
        let oz = self.orig.z.to_array();

        let dx = self.dir.x.to_array();
        let dy = self.dir.y.to_array();
        let dz = self.dir.z.to_array();

        let origin = Point3::new(
            ox[lane] as f64,
            oy[lane] as f64,
            oz[lane] as f64,
        );
        let direction = Vec3::new(
            dx[lane] as f64,
            dy[lane] as f64,
            dz[lane] as f64,
        );

        Ray::new(origin, direction)
    }

    /// Converts this `Ray4` pack into an array of 4 scalar `Ray`s.
    ///
    /// This is only used in debug / tests to compare the SIMD path
    /// against the scalar reference implementation.
    pub fn to_rays(&self) -> [Ray; 4] {
        [
            self.lane(0),
            self.lane(1),
            self.lane(2),
            self.lane(3),
        ]
    }
    
}

/// Construye un bloque de 4 rayos primarios para una fila `j` y un bloque de columnas
/// que comienza en `i_block`.
///
/// - Se generan hasta 4 rayos con la lógica estándar de la cámara (`get_ray`),
///   uno por cada píxel `(i_block + lane, j)`.
/// - Se maneja el caso en que el ancho de la imagen no sea múltiplo de 4 mediante
///   una máscara de lanes (`lane_valid`), donde:
///   - `lane_valid[lane] == true` si `i_block + lane < image_width`.
///   - `lane_valid[lane] == false` en caso contrario, y el rayo correspondiente
///     se deja en un valor por defecto.
/// - Se empaquetan los 4 rayos escalares en un `Ray4` para uso SIMD,
///   mientras que el arreglo `[Ray; 4]` se devuelve para el sombreado escalar.
pub fn build_primary_rays_block(
    camera: &Camera,
    j: i32,
    i_block: i32,
    image_width: i32,
) -> ([Ray; 4], Ray4, [bool; 4]) {
    // Se inicializa un arreglo de 4 rayos con valores por defecto.
    // Estos valores se utilizan para los lanes que queden fuera de rango.
    let mut rays = [
        Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)),
        Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)),
        Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)),
        Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)),
    ];

    // Se marca para cada lane si el píxel correspondiente cae dentro del ancho
    // real de la imagen. Esto permite manejar la última columna cuando el ancho
    // no es múltiplo de 4.
    let mut lane_valid = [false; 4];

    for lane in 0..4 {
        let ix = i_block + lane as i32;

        // Si la columna se sale del ancho, se deja el rayo por defecto y
        // se marca el lane como inválido.
        if ix >= image_width {
            lane_valid[lane] = false;
            continue;
        }

        // Se marca el lane como válido y se genera el rayo escalar estándar
        // para el píxel (ix, j).
        lane_valid[lane] = true;
        rays[lane] = camera.get_ray(ix, j);
    }

    // Se empaquetan los 4 rayos escalares en un Ray4 para uso SIMD
    // en la etapa de intersecciones.
    let ray4 = Ray4::from_rays(rays);

    (rays, ray4, lane_valid)
}
