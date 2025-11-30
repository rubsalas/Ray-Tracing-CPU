// src/simd/mod.rs

// Módulo raíz para utilidades SIMD del proyecto.
// Por ahora solo soporta NEON en aarch64.

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_tests;

#[cfg(target_arch = "aarch64")]
pub mod hit;
