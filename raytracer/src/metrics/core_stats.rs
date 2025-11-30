// src/metrics/core_stats.rs

use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Contadores centrales del pipeline de ray tracing.
///
/// Se agrupan métricas lógicas (no de hardware) para poder
/// comparar backends y escenas de forma consistente.
#[derive(Debug, Default, Clone, Copy)]
pub struct CoreStats {
    /// Número total de rayos primarios emitidos
    /// (uno por muestra y por píxel válido).
    pub rays_primary: u64,

    /// Número total de tests de intersección escalares
    /// (por ejemplo, Sphere::hit).
    pub scalar_intersection_tests: u64,
    /// Número de tests escalares que terminan en hit.
    pub scalar_intersection_hits: u64,
    /// Número de tests escalares que terminan en miss.
    pub scalar_intersection_misses: u64,

    /// Número total de tests de intersección vectorizados
    /// en world_hit4_spheres.
    pub simd_intersection_tests: u64,
    /// Número de lanes que presentan hit en SIMD.
    pub simd_intersection_hits_lanes: u64,
    /// Número de lanes que presentan miss en SIMD.
    pub simd_intersection_misses_lanes: u64,

    /// Número de llamadas a materiales Lambertian.
    pub lambertian_calls: u64,
    /// Número de llamadas a materiales Metal.
    pub metal_calls: u64,
    /// Número de llamadas a materiales Dielectric.
    pub dielectric_calls: u64,
    /// Número de veces que Dielectric produce reflexión.
    pub dielectric_reflect: u64,
    /// Número de veces que Dielectric produce refracción.
    pub dielectric_refract: u64,
}

static GLOBAL_CORE_STATS: Lazy<Mutex<CoreStats>> =
    Lazy::new(|| Mutex::new(CoreStats::default()));

/// Resetea todos los contadores a cero antes de un render.
pub fn reset_core_stats() {
    if let Ok(mut stats) = GLOBAL_CORE_STATS.lock() {
        *stats = CoreStats::default();
    }
}

/// Ejecuta una función con acceso mutable a los contadores.
pub fn with_core_stats<F, R>(f: F) -> R
where
    F: FnOnce(&mut CoreStats) -> R,
{
    let mut guard = GLOBAL_CORE_STATS
        .lock()
        .expect("se debe tomar el lock de GLOBAL_CORE_STATS");
    f(&mut *guard)
}

/// Obtiene una copia de los contadores actuales.
pub fn snapshot_core_stats() -> CoreStats {
    if let Ok(stats) = GLOBAL_CORE_STATS.lock() {
        *stats
    } else {
        CoreStats::default()
    }
}
