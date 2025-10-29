//! `camera` module
//!
//! Minimal pinhole camera that renders a scene by emitting one primary ray
//! per pixel
//!
//! Public knobs:
//! - [`Camera::aspect_ratio`] — image width over height
//! - [`Camera::image_width`]  — horizontal resolution in pixels
//!
//! Call [`Camera::render`] to write an ASCII PPM (`P3`) image to any `Write` sink.

use std::io::{Result as IoResult, Write};

use crate::interval::Interval;
use crate::prelude::*; // brings: Vec3, Point3, Color, Ray, write_color_to, unit_vector, infinity, etc.

/// Simple pinhole camera that renders a gradient sky and surface normals.
pub struct Camera {
    /// Image width in pixels.
    pub image_width: i32,
    /// Aspect ratio `width / height`.
    pub aspect_ratio: f64,

    // --- Derived / private state (filled by initialize) ---
    image_height: i32,
    center: Point3,
    pixel00_loc: Point3,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            image_width: 100,
            aspect_ratio: 1.0,
            image_height: 0,
            center: Point3::new(0.0, 0.0, 0.0),
            pixel00_loc: Point3::new(0.0, 0.0, 0.0),
            pixel_delta_u: Vec3::new(0.0, 0.0, 0.0),
            pixel_delta_v: Vec3::new(0.0, 0.0, 0.0),
        }
    }
}

impl Camera {
    /// Creates a camera with given `image_width` and `aspect_ratio`.
    pub fn new(image_width: i32, aspect_ratio: f64) -> Self {
        Self {
            image_width,
            aspect_ratio,
            ..Default::default()
        }
    }

    /// Renders the `world` to a PPM stream (`P3`)
    ///
    /// This function writes the header and pixel data to `out`. It computes the
    /// viewport, pixel deltas, and fires one primary ray per pixel. The pixel
    /// color is given by [`Camera::ray_color`].
    pub fn render<W: Write>(&mut self, world: &impl Hittable, out: &mut W) -> IoResult<()> {
        self.initialize();

        // PPM header
        writeln!(out, "P3")?;
        writeln!(out, "{} {}", self.image_width, self.image_height)?;
        writeln!(out, "255")?;

        for j in 0..self.image_height {
            eprint!("\rScanlines remaining: {} ", self.image_height - j);
            std::io::stderr().flush().ok();

            for i in 0..self.image_width {
                let pixel_center =
                    self.pixel00_loc + (i as f64) * self.pixel_delta_u + (j as f64) * self.pixel_delta_v;
                let ray_direction = pixel_center - self.center;
                let r = Ray::new(self.center, ray_direction);

                let pixel_color = self.ray_color(&r, world);
                write_color_to(out, pixel_color)?;
            }
        }

        eprintln!("\rDone.                 ");
        Ok(())
    }

    // --- Internals ---

    /// Computes derived camera parameters (viewport and pixel geometry).
    fn initialize(&mut self) {
        // Height from aspect ratio; clamp to at least 1
        self.image_height = (self.image_width as f64 / self.aspect_ratio) as i32;
        if self.image_height < 1 {
            self.image_height = 1;
        }

        self.center = Point3::new(0.0, 0.0, 0.0);

        // Viewport dimensions
        let focal_length = 1.0;
        let viewport_height = 2.0;
        let viewport_width = viewport_height * (self.image_width as f64 / self.image_height as f64);

        // Basis across the viewport edges
        let viewport_u = Vec3::new(viewport_width, 0.0, 0.0);
        let viewport_v = Vec3::new(0.0, -viewport_height, 0.0);

        // Per-pixel offsets
        self.pixel_delta_u = viewport_u / self.image_width as f64;
        self.pixel_delta_v = viewport_v / self.image_height as f64;

        // Upper-left pixel center
        let viewport_upper_left =
            self.center - Vec3::new(0.0, 0.0, focal_length) - viewport_u / 2.0 - viewport_v / 2.0;
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);
    }

    /// Background and surface-normal shading, identical to Listing 30 but using
    /// an [`Interval`] for the ray `t` range as introduced earlier.
    fn ray_color(&self, r: &Ray, world: &impl Hittable) -> Color {
        let mut rec = HitRecord::default();
        
        if world.hit(r, &Interval::new(0.0, INFINITY), &mut rec) {
            return 0.5 * (rec.normal + Color::new(1.0, 1.0, 1.0));
        }

        let unit_direction = unit_vector(r.direction());
        let a = 0.5 * (unit_direction.y + 1.0);
        (1.0 - a) * Color::new(1.0, 1.0, 1.0) + a * Color::new(0.5, 0.7, 1.0)
    }
}
