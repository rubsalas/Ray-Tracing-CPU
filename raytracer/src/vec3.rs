// src/vec3.rs
// Vec3 para RTIOW en Rust: operaciones básicas, indexación y alias útiles.
//
// - Campos públicos (x, y, z) para ergonomía
// - Operadores: +, -, *, /, negación;  v*v (component-wise), v*t y t*v, v/t
// - Indexación: v[0]=x, v[1]=y, v[2]=z (lectura y escritura)
// - Utilidades: length, length_squared, dot, cross, unit_vector
// - Alias: Point3 (= Vec3), Color (= Vec3)

use crate::prelude::random_double_range;

use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};
use rand::{distributions::Uniform, Rng};

#[derive(Copy, Clone, Default, PartialEq, Debug)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline]
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Returns `true` if the vector is close to zero in all components.
    ///
    /// Uses an absolute epsilon of `1e-8` on each axis.
    #[inline]
    pub fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && self.y.abs() < s && self.z.abs() < s
    }

    /// Returns a vector with each component sampled uniformly in [0.0, 1.0).
    #[inline]
    pub fn random() -> Self {
        Self::new(
            rand::random::<f64>(),
            rand::random::<f64>(),
            rand::random::<f64>(),
        )
    }

    /// Returns a vector with each component sampled uniformly in [min, max).
    #[inline]
    pub fn random_range(min: f64, max: f64) -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(min, max);
        Self::new(rng.sample(dist), rng.sample(dist), rng.sample(dist))
    }

    // Métodos asociados para quien prefiera estilo Vec3::dot(a,b)
    #[inline]
    pub fn dot(u: Self, v: Self) -> f64 {
        u.x * v.x + u.y * v.y + u.z * v.z
    }

    #[inline]
    pub fn cross(u: Self, v: Self) -> Self {
        Self {
            x: u.y * v.z - u.z * v.y,
            y: u.z * v.x - u.x * v.z,
            z: u.x * v.y - u.y * v.x,
        }
    }

    #[inline]
    pub fn unit_vector(v: Self) -> Self {
        let len = v.length();
        if len == 0.0 { v } else { v / len }
    }

    /// Samples a uniform random unit vector on the surface of the unit sphere.
    ///
    /// Rejection-samples a point `p` in the unit ball and normalizes it.
    /// Uses an epsilon to avoid dividing by ~0 when `p` is extremely small.
    ///
    /// - Draw `p ~ U([-1,1]^3)`
    /// - If `1e-160 < |p|^2 <= 1`, return `p / |p|`
    #[inline]
    pub fn random_unit_vector() -> Self {
        loop {
            let p = Self::random_range(-1.0, 1.0);
            let lensq = p.length_squared();
            if lensq > 1e-160 && lensq <= 1.0 {
                return p / lensq.sqrt();
            }
        }
    }

    /// Samples a random unit vector on the hemisphere defined by `normal`.
    ///
    /// Draws a uniform random direction on the unit sphere and flips it if
    /// it falls on the opposite hemisphere relative to `normal`.
    #[inline]
    pub fn random_on_hemisphere(normal: Vec3) -> Self {
        let on_unit_sphere = Self::random_unit_vector();
        if dot(on_unit_sphere, normal) > 0.0 {
            on_unit_sphere // same hemisphere as `normal`
        } else {
            -on_unit_sphere // flip to match hemisphere
        }
    }

}

// -------- Indexación: v[i] <-> x/y/z --------
impl Index<usize> for Vec3 {
    type Output = f64;
    fn index(&self, i: usize) -> &Self::Output {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of range"),
        }
    }
}
impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index out of range"),
        }
    }
}

// -------- Operadores unarios/binarios --------
impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output { Self { x: -self.x, y: -self.y, z: -self.z } }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x; self.y += rhs.y; self.z += rhs.z;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x; self.y -= rhs.y; self.z -= rhs.z;
    }
}

// Producto componente a componente (útil para colores)
impl Mul for Vec3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self { x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z }
    }
}

// Escalar a la derecha: v * t
impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, t: f64) -> Self::Output {
        Self { x: self.x * t, y: self.y * t, z: self.z * t }
    }
}
impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, t: f64) {
        self.x *= t; self.y *= t; self.z *= t;
    }
}

// Escalar a la izquierda: t * v
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Self::Output {
        Vec3 { x: self * v.x, y: self * v.y, z: self * v.z }
    }
}

// División por escalar: v / t
impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, t: f64) -> Self::Output { (1.0 / t) * self }
}
impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, t: f64) {
        let inv = 1.0 / t;
        self.x *= inv; self.y *= inv; self.z *= inv;
    }
}

// -------- Funciones libres --------
#[inline] pub fn dot(u: Vec3, v: Vec3) -> f64 { Vec3::dot(u, v) }
#[inline] pub fn cross(u: Vec3, v: Vec3) -> Vec3 { Vec3::cross(u, v) }
#[inline] pub fn unit_vector(v: Vec3) -> Vec3 { Vec3::unit_vector(v) }

/// Reflects a vector `v` about a surface normal `n`.
///
/// Formula: `v - 2 * dot(v, n) * n`
///
/// Assumes `n` is a unit vector for correct geometric reflection.
///
/// # Example
/// ```rust
/// # use crate::vec3::{Vec3, reflect, unit_vector, dot};
/// let v = Vec3::new(1.0, -1.0, 0.0);
/// let n = unit_vector(Vec3::new(0.0, 1.0, 0.0)); // y-up
/// let r = reflect(v, n);
/// // r should be (1, 1, 0)
/// assert!((r.x - 1.0).abs() < 1e-12 && (r.y - 1.0).abs() < 1e-12 && (r.z).abs() < 1e-12);
/// ```
#[inline]
pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * dot(v, n) * n
}

/// Refracts an incoming unit direction `uv` through a surface with normal `n`
/// using Snell's law. `etai_over_etat` is the ratio ηᵢ/ηₜ (from incident to transmitted).
///
/// Assumes:
/// - `uv` is unit-length (normalize before calling).
/// - `n` is a unit normal, oriented to face the incident ray.
///
/// Formula:
/// ```text
/// cos_theta      = min(dot(-uv, n), 1)
/// r_out_perp     = etai_over_etat * (uv + cos_theta * n)
/// r_out_parallel = -sqrt(|1 - |r_out_perp|^2|) * n
/// refracted      = r_out_perp + r_out_parallel
/// ```
///
/// Returns the refracted direction (not necessarily normalized, but typically close).
#[inline]
pub fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = f64::min(dot(-uv, n), 1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -((1.0 - r_out_perp.length_squared()).abs().sqrt()) * n;
    r_out_perp + r_out_parallel
}

/// Returns a random point uniformly distributed inside the unit disk on the XY plane.
/// Uses rejection sampling in the square [-1, +1]² until `|p|² < 1`.
#[inline]
pub fn random_in_unit_disk() -> Vec3 {
    loop {
        let p = Vec3::new(
            random_double_range(-1.0, 1.0),
            random_double_range(-1.0, 1.0),
            0.0,
        );
        if p.length_squared() < 1.0 {
            return p;
        }
    }
}

// -------- Alias semanticos --------
pub type Point3 = Vec3; // puntos 3D
pub type Color  = Vec3; // colores RGB
