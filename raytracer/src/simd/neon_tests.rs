// src/simd/neon_tests.rs

use super::neon::*;          // F32x4, Vec3x4, Ray4, build_primary_rays_block, etc.

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
    use crate::ray::Ray;
    use crate::camera::Camera;
    use crate::vec3::{Point3, Vec3};

    /// Simple helper for approximate floating-point comparisons.
    ///
    /// Returns `true` if the absolute difference between `a` and `b`
    /// is less than or equal to `eps`.
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn approx_eq_f64(a: f64, b: f64, eps: f64) -> bool {
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

    #[test]
    fn ray4_from_rays_and_lane_roundtrip() {
        // Four simple, distinct rays so that we can verify each lane.
        let r0 = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let r1 = Ray::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 1.0, 0.0));
        let r2 = Ray::new(Point3::new(-1.0, -2.0, -3.0), Vec3::new(0.0, 0.0, 1.0));
        let r3 = Ray::new(Point3::new(10.0, 20.0, 30.0), Vec3::new(1.0, 1.0, 1.0));

        let rays = [r0, r1, r2, r3];

        // Pack the four rays into Ray4.
        let ray4 = Ray4::from_rays(rays);

        // Unpack each lane and compare back to the original rays.
        for lane in 0..4 {
            let original = &rays[lane];
            let unpacked = ray4.lane(lane);

            let o_orig = original.origin();
            let d_orig = original.direction();

            let o_unpack = unpacked.origin();
            let d_unpack = unpacked.direction();

            // Allow a small epsilon due to f64 -> f32 -> f64 conversion.
            let eps = 1e-5;

            assert!(
                approx_eq_f64(o_orig.x, o_unpack.x, eps)
                    && approx_eq_f64(o_orig.y, o_unpack.y, eps)
                    && approx_eq_f64(o_orig.z, o_unpack.z, eps),
                "Origin mismatch at lane {}: orig={:?}, unpack={:?}",
                lane,
                o_orig,
                o_unpack
            );

            assert!(
                approx_eq_f64(d_orig.x, d_unpack.x, eps)
                    && approx_eq_f64(d_orig.y, d_unpack.y, eps)
                    && approx_eq_f64(d_orig.z, d_unpack.z, eps),
                "Direction mismatch at lane {}: orig={:?}, unpack={:?}",
                lane,
                d_orig,
                d_unpack
            );
        }
    }

    /// Verifies that the SIMD computation
    ///   sample = p00 + u * du + v * dv
    /// matches the scalar computation lane by lane
    /// for a simple synthetic case, using pure f32 math.
    #[test]
    fn camera_geometry_block_matches_scalar() {
        // Synthetic camera geometry in f32:
        // p00 = (1, 2, 3)
        // du  = (0.5, 0.0, -0.5)
        // dv  = (0.0, 1.0,  0.25)
        let p00 = [1.0_f32, 2.0_f32, 3.0_f32];
        let du  = [0.5_f32, 0.0_f32, -0.5_f32];
        let dv  = [0.0_f32, 1.0_f32, 0.25_f32];

        // Four lanes of u/v (simulating four different pixels).
        let u = [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32];
        let v = [0.0_f32, 0.5_f32, 1.0_f32, 1.5_f32];

        // Scalar reference: sample_scalar[lane] = p00 + u[lane]*du + v[lane]*dv
        let mut scalar_samples = [[0.0_f32; 3]; 4];
        for lane in 0..4 {
            let ux = u[lane];
            let vx = v[lane];

            scalar_samples[lane][0] = p00[0] + ux * du[0] + vx * dv[0];
            scalar_samples[lane][1] = p00[1] + ux * du[1] + vx * dv[1];
            scalar_samples[lane][2] = p00[2] + ux * du[2] + vx * dv[2];
        }

        // SIMD path: mimic the math used in Camera::get_ray4_from_indices.
        let p00x = F32x4::splat(p00[0]);
        let p00y = F32x4::splat(p00[1]);
        let p00z = F32x4::splat(p00[2]);

        let dux = F32x4::splat(du[0]);
        let duy = F32x4::splat(du[1]);
        let duz = F32x4::splat(du[2]);

        let dvx = F32x4::splat(dv[0]);
        let dvy = F32x4::splat(dv[1]);
        let dvz = F32x4::splat(dv[2]);

        let u4 = F32x4::from_array(u);
        let v4 = F32x4::from_array(v);

        let sample_x = p00x.add(dux.mul(u4)).add(dvx.mul(v4));
        let sample_y = p00y.add(duy.mul(u4)).add(dvy.mul(v4));
        let sample_z = p00z.add(duz.mul(u4)).add(dvz.mul(v4));

        let sx = sample_x.to_array();
        let sy = sample_y.to_array();
        let sz = sample_z.to_array();

        let eps = 1e-5_f32;

        for lane in 0..4 {
            let s = scalar_samples[lane];

            assert!(
                approx_eq(s[0], sx[lane], eps)
                    && approx_eq(s[1], sy[lane], eps)
                    && approx_eq(s[2], sz[lane], eps),
                "Mismatch at lane {}: scalar=({:.6}, {:.6}, {:.6}), simd=({:.6}, {:.6}, {:.6})",
                lane,
                s[0], s[1], s[2],
                sx[lane], sy[lane], sz[lane],
            );
        }
    }

    #[test]
    fn build_primary_rays_block_marks_valid_lanes_on_edge() {
        // Se crea una cámara con parámetros sencillos.
        let cam = Camera::new(8, 16.0 / 9.0);
        let j = 0;
        let image_width = 8;

        // Se escoge un bloque que empieza en la columna 6:
        // lanes:
        //   lane 0 -> x = 6 (válido)
        //   lane 1 -> x = 7 (válido)
        //   lane 2 -> x = 8 (inválido)
        //   lane 3 -> x = 9 (inválido)
        let i_block = 6;

        let (rays, _ray4, lane_valid) =
            build_primary_rays_block(&cam, j, i_block, image_width);

        // Se verifica la máscara de lanes.
        assert!(lane_valid[0], "lane 0 debe ser válido (x = 6)");
        assert!(lane_valid[1], "lane 1 debe ser válido (x = 7)");
        assert!(!lane_valid[2], "lane 2 debe ser inválido (x = 8 >= width)");
        assert!(!lane_valid[3], "lane 3 debe ser inválido (x = 9 >= width)");

        // Se verifica que para lanes inválidos se mantenga el rayo por defecto.
        // Como se inicializa con origen y dirección (0,0,0), se utiliza eso
        // como condición de que no se ha modificado.
        for lane in 2..4 {
            let r = &rays[lane];
            let o = r.origin();
            let d = r.direction();
            assert!(
                o.x == 0.0 && o.y == 0.0 && o.z == 0.0,
                "lane {} inválido debe conservar origen por defecto", lane
            );
            assert!(
                d.x == 0.0 && d.y == 0.0 && d.z == 0.0,
                "lane {} inválido debe conservar dirección por defecto", lane
            );
        }
    }

    #[test]
    fn build_primary_rays_block_marks_all_lanes_valid_when_inside_width() {
        let cam = Camera::new(8, 16.0 / 9.0);
        let j = 1;
        let image_width = 8;

        // Se escoge un bloque que empieza en la columna 2:
        // lanes:
        //   lane 0 -> x = 2 (válido)
        //   lane 1 -> x = 3 (válido)
        //   lane 2 -> x = 4 (válido)
        //   lane 3 -> x = 5 (válido)
        let i_block = 2;

        let (_rays, _ray4, lane_valid) =
            build_primary_rays_block(&cam, j, i_block, image_width);

        for lane in 0..4 {
            assert!(
                lane_valid[lane],
                "lane {} debe ser válido dentro del ancho de la imagen",
                lane
            );
        }
    }

}
