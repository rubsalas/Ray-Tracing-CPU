//! Camera module
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

use crate::prelude::*;

#[cfg(target_arch = "aarch64")]
use crate::simd::neon::{F32x4, Ray4, Vec3x4};

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

    // Depth of field
    /// Variation angle of rays through each pixel (degrees). 0 disables DoF.
    pub defocus_angle: f64,
    /// Distance from lookfrom to plane of perfect focus.
    pub focus_dist: f64,

    // Derived / private state (filled by initialize)
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

    // Defocus disk radii (world-space)
    defocus_disk_u: Vec3,
    defocus_disk_v: Vec3,
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

            defocus_angle: 0.0,
            focus_dist: 10.0,

            image_height: 0,
            pixel_samples_scale: 0.0,
            center: Point3::new(0.0, 0.0, 0.0),
            pixel00_loc: Point3::new(0.0, 0.0, 0.0),
            pixel_delta_u: Vec3::new(0.0, 0.0, 0.0),
            pixel_delta_v: Vec3::new(0.0, 0.0, 0.0),

            u: Vec3::new(0.0, 0.0, 0.0),
            v: Vec3::new(0.0, 0.0, 0.0),
            w: Vec3::new(0.0, 0.0, 0.0),

            defocus_disk_u: Vec3::new(0.0, 0.0, 0.0),
            defocus_disk_v: Vec3::new(0.0, 0.0, 0.0),
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
    pub fn initialize(&mut self) {
        // Image geometry
        self.image_height = (self.image_width as f64 / self.aspect_ratio) as i32;
        if self.image_height < 1 { self.image_height = 1; }

        self.pixel_samples_scale = 1.0 / (self.samples_per_pixel as f64);
        self.center = self.lookfrom;

        // Viewport size from vfov and focus distance
        let theta = degrees_to_radians(self.vfov);
        let h = (theta * 0.5).tan();
        let viewport_height = 2.0 * h * self.focus_dist;
        let viewport_width  = viewport_height * (self.image_width as f64 / self.image_height as f64);

        // Camera basis
        self.w = unit_vector(self.lookfrom - self.lookat);
        self.u = unit_vector(cross(self.vup, self.w));
        self.v = cross(self.w, self.u);

        // Viewport edges
        let viewport_u = viewport_width * self.u;      // across
        let viewport_v = viewport_height * -self.v;    // down

        // Pixel deltas
        self.pixel_delta_u = viewport_u / self.image_width as f64;
        self.pixel_delta_v = viewport_v / self.image_height as f64;

        // Upper-left pixel center
        let viewport_upper_left = self.center
                                 - self.focus_dist * self.w
                                 - 0.5 * viewport_u
                                 - 0.5 * viewport_v;
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);

        // Defocus disk radii (world space); zero if defocus_angle == 0
        let defocus_radius = self.focus_dist * (degrees_to_radians(self.defocus_angle * 0.5)).tan();
        self.defocus_disk_u = self.u * defocus_radius;
        self.defocus_disk_v = self.v * defocus_radius;
    }

    /// Returns a random offset inside the pixel: [-0.5, +0.5]^2 (z = 0).
    #[inline]
    fn sample_square(&self) -> Vec3 {
        Vec3::new(random_double() - 0.5, random_double() - 0.5, 0.0)
    }

    /// Sample a random point on the defocus disk (world space).
    #[inline]
    fn defocus_disk_sample(&self) -> Point3 {
        let p = random_in_unit_disk(); // XY unit disk
        self.center + p.x * self.defocus_disk_u + p.y * self.defocus_disk_v
    }

    /// Ray through a jittered sample in pixel (i, j), originating at center or defocus disk.
    #[inline]
    pub fn get_ray(&self, i: i32, j: i32) -> Ray {
        let offset = self.sample_square();
        let pixel_sample = self.pixel00_loc
            + (i as f64 + offset.x) * self.pixel_delta_u
            + (j as f64 + offset.y) * self.pixel_delta_v;

        let ray_origin =
            if self.defocus_angle <= 0.0 { self.center } else { self.defocus_disk_sample() };
        let ray_direction = pixel_sample - ray_origin;

        Ray::new(ray_origin, ray_direction)
    }

    /// Renders `world` to a PPM stream (`P3`) with multi-sampling.
    #[allow(dead_code)]
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

    /// Ray color
    pub fn ray_color(&self, r: &Ray, depth: i32, world: &dyn Hittable) -> Color {
        // If the ray bounce limit is exceeded, no more light is gathered.
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

    /// Con esto el backend puede preguntar “cuál fue la altura final”
    /// y “en cuánto escalo el color por 1/spp”.
    pub fn image_height(&self) -> i32 {
        self.image_height
    }

    pub fn pixel_samples_scale(&self) -> f64 {
        self.pixel_samples_scale
    }  

    /// Builds four rays for pixels `(i_base + 0, i_base + 1, i_base + 2, i_base + 3)`
    /// on the same scanline `j`, using:
    ///
    /// - Scalar randomness for per-pixel jitter and (optionally) depth of field.
    /// - SIMD math (`F32x4` + `Vec3x4`) for camera geometry in world-space.
    ///
    /// The Neon backend uses this method to construct 4 rays in parallel,
    /// and then unpacks each lane back into a scalar `Ray` to feed the existing
    /// `ray_color` implementation. This keeps shading fully scalar in phase 4,
    /// while already exercising the SIMD camera path.
    ///
    /// High-level steps:
    /// 1) For each lane 0..3:
    ///    - Sample a random offset inside the pixel via `sample_square()`.
    ///    - Build the `(u, v)` coordinates for that sample.
    /// 2) Pack the four `u` and four `v` into `F32x4` vectors (`u4`, `v4`).
    /// 3) Compute the four pixel sample positions:
    ///      `pixel_sample = pixel00_loc + u * pixel_delta_u + v * pixel_delta_v`
    ///    in parallel for x, y, z.
    /// 4) Build four ray origins:
    ///    - If defocus is disabled, all origins are `center`.
    ///    - If defocus is enabled, each lane gets its own sample in the defocus disk.
    /// 5) Directions are computed as `pixel_sample - origin` in SIMD.
    /// 6) The result is returned as `Ray4 { orig: Vec3x4, dir: Vec3x4 }`.
    #[cfg(target_arch = "aarch64")]
    pub fn get_ray4_from_indices(&self, i_base: i32, j: i32) -> Ray4 {
        // ----------------------------
        // 1) Scalar jitter per lane → u[], v[]
        // ----------------------------

        // Arrays of 4 lanes for pixel sample coordinates in image space.
        // We keep them as f32 because the SIMD layer works in f32 (NEON float32x4).
        let mut u = [0.0f32; 4];
        let mut v = [0.0f32; 4];

        for lane in 0..4 {
            // Original scalar path uses `sample_square()` to jitter inside the pixel:
            // offset.x, offset.y in [-0.5, +0.5].
            let offset = self.sample_square();

            let i_f = (i_base + lane as i32) as f64 + offset.x;
            let j_f = j as f64 + offset.y;

            u[lane] = i_f as f32;
            v[lane] = j_f as f32;
        }

        // Pack u[] and v[] into NEON vectors (4 lanes each).
        let u4 = F32x4::from_array(u);
        let v4 = F32x4::from_array(v);

        // ----------------------------
        // 2) Camera geometry in SIMD form
        // ----------------------------
        //
        // We reuse the internal camera state computed in `initialize()`:
        // - pixel00_loc: upper-left pixel center in world-space
        // - pixel_delta_u: step in world-space when moving +1 in image x
        // - pixel_delta_v: step in world-space when moving +1 in image y

        let p00 = &self.pixel00_loc;
        let du = &self.pixel_delta_u;
        let dv = &self.pixel_delta_v;

        // Broadcast the world-space base point and deltas into SIMD lanes.
        let p00x = F32x4::splat(p00.x as f32);
        let p00y = F32x4::splat(p00.y as f32);
        let p00z = F32x4::splat(p00.z as f32);

        let dux = F32x4::splat(du.x as f32);
        let duy = F32x4::splat(du.y as f32);
        let duz = F32x4::splat(du.z as f32);

        let dvx = F32x4::splat(dv.x as f32);
        let dvy = F32x4::splat(dv.y as f32);
        let dvz = F32x4::splat(dv.z as f32);

        // pixel_sample = pixel00_loc + u * pixel_delta_u + v * pixel_delta_v
        //
        // We do this component-wise using F32x4 arithmetic. The exact
        // combination uses the helper methods `mul` and `add` we already
        // defined in F32x4.
        let sample_x = p00x.add(dux.mul(u4)).add(dvx.mul(v4));
        let sample_y = p00y.add(duy.mul(u4)).add(dvy.mul(v4));
        let sample_z = p00z.add(duz.mul(u4)).add(dvz.mul(v4));

        // ----------------------------
        // 3) Ray origins (with or without defocus)
        // ----------------------------
        //
        // If defocus is disabled, all rays originate at `center`.
        // If defocus is enabled, we sample one point in the defocus disk per lane.

        let mut ox = [0.0f32; 4];
        let mut oy = [0.0f32; 4];
        let mut oz = [0.0f32; 4];

        if self.defocus_angle <= 0.0 {
            // No depth-of-field: all origins equal to camera center.
            for lane in 0..4 {
                ox[lane] = self.center.x as f32;
                oy[lane] = self.center.y as f32;
                oz[lane] = self.center.z as f32;
            }
        } else {
            // Depth-of-field enabled: per-lane random point on defocus disk.
            for lane in 0..4 {
                let origin = self.defocus_disk_sample();
                ox[lane] = origin.x as f32;
                oy[lane] = origin.y as f32;
                oz[lane] = origin.z as f32;
            }
        }

        let origin_x = F32x4::from_array(ox);
        let origin_y = F32x4::from_array(oy);
        let origin_z = F32x4::from_array(oz);

        // ----------------------------
        // 4) Directions = sample - origin
        // ----------------------------

        let dir_x = sample_x.sub(origin_x);
        let dir_y = sample_y.sub(origin_y);
        let dir_z = sample_z.sub(origin_z);

        // ----------------------------
        // 5) Pack into Vec3x4 and Ray4
        // ----------------------------

        let orig = Vec3x4 {
            x: origin_x,
            y: origin_y,
            z: origin_z,
        };

        let dir = Vec3x4 {
            x: dir_x,
            y: dir_y,
            z: dir_z,
        };

        Ray4 { orig, dir }
    }

}
