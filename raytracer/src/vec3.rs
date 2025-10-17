// src/vec3.rs

use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Vec3: vector 3D básico para puntos, direcciones y colores.
#[derive(Copy, Clone, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {

    /// Constructor: nuevo vector (x, y, z)
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    // --- Getters ---
    #[inline] pub fn x(&self) -> f64 { self.x }
    #[inline] pub fn y(&self) -> f64 { self.y }
    #[inline] pub fn z(&self) -> f64 { self.z }

    // --- Longitudes ---

    /// length_squared(): evita hacer sqrt cuando no es necesario.
    #[inline]
    pub fn length_squared(&self) -> f64 {
        (self.x * self.x) + (self.y * self.y) + (self.z * self.z) 
    }

    /// length(): norma L2 (sqrt de la suma de cuadrados).
    #[inline]
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    // --- Operaciones vectoriales ---

    /// Producto punto
    #[inline]
    pub fn dot(u: Self, v: Self) -> f64 {
        (u.x * v.x) + (u.y * v.y) + (u.z + v.z)
    }

    /// Producto cruz
    #[inline]
    pub fn cross(u: Self, v: Self) -> Self {
        Self::new(
            (u.y * v.z) - (u.z * v.y),
            (u.z * v.x) - (u.x * v.z),
            (u.x * v.y) - (u.y * v.x),
        )
    }

    /// unit_vector(): devuelve el vector normalizado
    /// Si la longitud es 0, se devuelve el mismo vector
    #[inline]
    pub fn unit_vector(v: Self) -> Self {
        let len = v.length();
        if len == 0.0 { v } else { v / len }
    }
    
}

// --- Implementaciones de formato para imprimir el vector
impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Formato legible con 6 decimales
        write!(f, "Vec3({:.6}, {:.6}, {:.6})", self.x, self.y, self.z)
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Imprime "x y z"
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}


// --- Indexing: v[0] -> x, v[1] -> y, v[2] -> z (read y write).

impl Index<usize> for Vec3 {
    type Output = f64;
    fn index(&self, i:usize) -> &Self::Output {
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

// --- Operadores ---

// -u
impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

// u + v
impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x; self.y += rhs.y; self.z += rhs.z;
    }
}

// u - v
impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x; self.y -= rhs.y; self.z -= rhs.z;
    }
}

// u * v (producto componente a componente)
impl Mul for Vec3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

// v * t  (escalar a la derecha)
impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, t: f64) -> Self::Output {
        Self::new(self.x * t, self.y * t, self.z * t)
    }
}
impl MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, t: f64) {
        self.x *= t; self.y *= t; self.z *= t;
    }
}

// t * v  (escalar a la izquierda)
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Self::Output {
        Vec3::new(self * v.x, self * v.y, self * v.z)
    }
}

// v / t (divide por escalar)
impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, t: f64) -> Self::Output {
        // Igual que (1/t) * v
        (1.0 / t) * self
    }
}
impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, t: f64) {
        let inv = 1.0 / t;
        self.x *= inv; self.y *= inv; self.z *= inv;
    }
}

// --- Funciones utilitarias ---


#[inline]
pub fn dot(u: Vec3, v: Vec3) -> f64 {
    Vec3::dot(u, v)
}

#[inline]
pub fn cross(u: Vec3, v: Vec3) -> Vec3 {
    Vec3::cross(u, v)
}

#[inline]
pub fn unit_vector(v: Vec3) -> Vec3 {
    Vec3::unit_vector(v)
}

// --- Alias geométrico: Point3 = Vec3 ---

pub type Point3 = Vec3;
pub type Color  = Vec3;
