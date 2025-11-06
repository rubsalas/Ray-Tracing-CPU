//! `hittable_list` module
//!
//! Minimal container for multiple [`Hittable`](crate::hittable::Hittable) objects.
//! Reports the closest intersection in a parametric interval by iterating
//! over all children.
//!
//! # Examples
//! Build a world with one sphere and query a hit:
//! ```rust,no_run
//! # use crate::vec3::{Point3, Vec3};
//! # use crate::rays::Ray;
//! # use crate::sphere::Sphere;
//! # use crate::hittable_list::{HittableList, HittablePtr};
//! # use std::rc::Rc;
//! let mut world = HittableList::new();
//! world.add(Rc::new(Sphere::new(Point3::new(0.0, 0.0, -1.0), 0.5)) as HittablePtr);
//!
//! let r = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
//! let mut rec = crate::hittable::HitRecord::default();
//! let hit = world.hit(&r, 1e-8, f64::INFINITY, &mut rec);
//! # let _ = hit;
//! ```

use std::rc::Rc;

use crate::ray::Ray;
use crate::interval::Interval;
use crate::hittable::{HitRecord, Hittable};

/// Shared reference type for hittables
pub type HittablePtr = Rc<dyn Hittable>;

/// A collection of shared hittable objects.
///
/// The list owns shared references (`Rc`) so the same object can be reused
/// across multiple containers without copying.
#[derive(Default)]
pub struct HittableList {
    objects: Vec<HittablePtr>,
}

impl HittableList {
    /// Creates an empty list.
    pub fn new() -> Self {
        Self { objects: Vec::new() }
    }

    /// Creates a list containing a single object.
    pub fn with(object: HittablePtr) -> Self {
        let mut list = Self::new();
        list.add(object);
        list
    }

    /// Removes all objects.
    pub fn clear(&mut self) {
        self.objects.clear();
    }

    /// Adds a hittable object by shared pointer.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use crate::vec3::Point3;
    /// # use crate::sphere::Sphere;
    /// # use crate::hittable_list::{HittableList, HittablePtr};
    /// # use std::rc::Rc;
    /// let mut world = HittableList::new();
    /// world.add(Rc::new(Sphere::new(Point3::new(0.0,0.0,-1.0), 0.5)) as HittablePtr);
    /// ```
    pub fn add(&mut self, object: HittablePtr) {
        self.objects.push(object);
    }

    /// Adds a hittable object given as a boxed trait object.
    ///
    /// Convenience for code that constructs `Box<dyn Hittable>`; converts to `Rc`.
    pub fn add_boxed(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(Rc::from(object));
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, ray_t: &Interval, rec: &mut HitRecord) -> bool {
        let mut temp_rec = HitRecord::default();
        let mut hit_anything = false;
        let mut closest_so_far = ray_t.max;

        for obj in &self.objects {
            let window = Interval::new(ray_t.min, closest_so_far);
            if obj.hit(r, &window, &mut temp_rec) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec.clone_from(&temp_rec); // ≈ *rec = temp_rec.clone();
            }
        }
        hit_anything
    }
}
