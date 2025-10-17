// src/ray.rs

use crate::vec3::{Point3, Vec3};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ray {
    /// Origen del rayo (punto 3D)
    orig: Point3,
    /// Dirección del rayo (vector 3D, se asume normalizado o no según el uso)
    dir: Vec3,
}

impl Ray {

    /// Crea un nuevo rayo con `origin` y `direction`.
    /// Constructor
    #[inline]
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Self { orig: origin, dir: direction }
    }

    /// Getter del origen
    /// Return por valor porque `Point3` es Copy.
    #[inline]
    pub fn origin(&self) -> Point3 {
        self.orig
    }

    /// Getter de la dirección
    #[inline]
    pub fn direction(&self) -> Vec3 {
        self.dir
    }

    /// `at(t) = origin + t * direction`
    /// Return el punto sobre la recta paramétrica a distancia `t` del origen.
    #[inline]
    pub fn at(&self, t: f64) -> Point3 {
        self.orig + t * self.dir
    }

}
