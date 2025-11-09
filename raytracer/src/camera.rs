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

use crate::prelude::*; // brings: Vec3, Point3, Color, Ray, write_color_to, unit_vector, infinity, etc.
// use crate::interval::Interval;

/// Simple pinhole camera that renders a gradient sky and surface normals.
pub struct Camera {
    /// Image width in pixels.
    pub image_width: i32,
    /// Aspect ratio `width / height`.
    pub aspect_ratio: f64,
    /// Number of random samples per pixel.
    pub samples_per_pixel: i32,
    /// Maximum number of ray bounces into scene
    pub max_depth: i32,
    /// Vertical field of view in degrees
    pub vfov: f64,

    /// Camera pose
    pub lookfrom: Point3,   // where the camera is
    pub lookat:   Point3,   // what the camera looks at
    pub vup:      Vec3,     // “up” direction

    // --- Derived / private state (filled by initialize) ---
    image_height: i32,
    pixel_samples_scale: f64,
    center: Point3,
    pixel00_loc: Point3,
    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,

    // Camera frame basis
    u: Vec3,
    v: Vec3,
    w: Vec3,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            image_width: 100,
            aspect_ratio: 1.0,
            samples_per_pixel: 10,
            max_depth: 10,
            vfov: 90.0,

            // Default pose
            lookfrom: Point3::new(0.0, 0.0, 0.0),
            lookat:   Point3::new(0.0, 0.0, -1.0),
            vup:      Vec3::new(0.0, 1.0, 0.0),

            image_height: 0,
            pixel_samples_scale: 0.0,
            center: Point3::new(0.0, 0.0, 0.0),
            pixel00_loc: Point3::new(0.0, 0.0, 0.0),
            pixel_delta_u: Vec3::new(0.0, 0.0, 0.0),
            pixel_delta_v: Vec3::new(0.0, 0.0, 0.0),

            u: Vec3::new(0.0, 0.0, 0.0),
            v: Vec3::new(0.0, 0.0, 0.0),
            w: Vec3::new(0.0, 0.0, 0.0),
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

    /// Computes derived camera parameters (viewport and pixel geometry).
    fn initialize(&mut self) {
        // Image height (≥ 1)
        self.image_height = (self.image_width as f64 / self.aspect_ratio) as i32;
        if self.image_height < 1 { self.image_height = 1; }

        self.pixel_samples_scale = 1.0 / (self.samples_per_pixel as f64);

        // Camera center
        self.center = self.lookfrom;

        // Viewport dimensions from vfov and focus distance
        let focal_length = (self.lookfrom - self.lookat).length();
        let theta = degrees_to_radians(self.vfov);
        let h = (theta * 0.5).tan();
        let viewport_height = 2.0 * h * focal_length;
        let viewport_width  =
            viewport_height * (self.image_width as f64 / self.image_height as f64);

        // Orthonormal basis (u, v, w)
        self.w = unit_vector(self.lookfrom - self.lookat);
        self.u = unit_vector(cross(self.vup, self.w));
        self.v = cross(self.w, self.u);

        // Edges of the viewport in world space
        let viewport_u = viewport_width * self.u;     // across (right)
        let viewport_v = viewport_height * -self.v;   // down

        // Pixel deltas
        self.pixel_delta_u = viewport_u / self.image_width as f64;
        self.pixel_delta_v = viewport_v / self.image_height as f64;

        // Upper-left pixel center
        let viewport_upper_left = self.center - (focal_length * self.w)
                                               - viewport_u * 0.5
                                               - viewport_v * 0.5;
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);
    }

    /// Renders `world` to a PPM stream (`P3`) with multi-sampling.
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
                // Sum of samples for this pixel
                let mut pixel_color = Color::new(0.0, 0.0, 0.0);
                for _s in 0..self.samples_per_pixel {
                    let r = self.get_ray(i, j);
                    pixel_color += self.ray_color(&r, self.max_depth, world);
                }
                // Scale by 1/spp before writing
                write_color_to(out, self.pixel_samples_scale * pixel_color)?;
            }
        }

        eprintln!("\rDone.                 ");
        Ok(())
    }

    /// Returns a random offset inside the unit square centered at the pixel: [-0.5, +0.5]^2 (z=0).
    #[inline]
    fn sample_square(&self) -> Vec3 {
        Vec3::new(random_double() - 0.5, random_double() - 0.5, 0.0)
    }

    /// Constructs a ray through a randomly jittered point around pixel `(i, j)`.
    ///
    /// Matches the intent of `get_ray` in Listing 45: sample inside the unit
    /// square around the pixel center via [`sample_square`], convert that to
    /// world-space using `pixel_delta_u/v`, and shoot from `center`.
    fn get_ray(&self, i: i32, j: i32) -> Ray {
        let offset = self.sample_square(); // in [-0.5, +0.5]²
        let pixel_sample = self.pixel00_loc
            + (i as f64 + offset.x) * self.pixel_delta_u
            + (j as f64 + offset.y) * self.pixel_delta_v;

        let ray_origin = self.center;
        let ray_direction = pixel_sample - ray_origin;
        Ray::new(ray_origin, ray_direction)
    }

    /// Ray color
    fn ray_color(&self, r: &Ray, depth: i32, world: &impl Hittable) -> Color {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if depth <= 0 {
            return Color::new(0.0, 0.0, 0.0);
        }
    
        let mut rec = HitRecord::default();
    
        if world.hit(r, &Interval::new(0.001, INFINITY), &mut rec) {
            if let Some(mat) = &rec.mat {
                let mut scattered = Ray::new(rec.p, Vec3::new(0.0, 0.0, 0.0)); // will be overwritten by scatter
                let mut attenuation = Color::new(0.0, 0.0, 0.0);
    
                if mat.scatter(r, &rec, &mut attenuation, &mut scattered) {
                    return attenuation * self.ray_color(&scattered, depth - 1, world);
                }
            }
            // Absorbed (no scatter or missing material)
            return Color::new(0.0, 0.0, 0.0);
        }
    
        // Background gradient
        let unit_direction = unit_vector(r.direction());
        let a = 0.5 * (unit_direction.y + 1.0);
        (1.0 - a) * Color::new(1.0, 1.0, 1.0) + a * Color::new(0.5, 0.7, 1.0)
    }
}
