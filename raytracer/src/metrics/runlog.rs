// src/metrics/runlog.rs

use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Result as IoResult, Write};
use std::path::Path;

use crate::metrics::RunMetrics;

pub fn write_metrics_to_file(path: &Path, metrics: &RunMetrics) -> IoResult<()> {
    // Se crea el directorio padre si no existe.
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }

    // Se abre el archivo de salida en modo escritura con buffer.
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    // ──────────────────────────────────────────────────────────────
    // Encabezado general de la corrida
    // ──────────────────────────────────────────────────────────────
    writeln!(out, "run_id: {}", metrics.run_id)?;
    writeln!(out, "backend: {:?}", metrics.backend)?;
    writeln!(out, "image_width: {}", metrics.image_width)?;
    writeln!(out, "image_height: {}", metrics.image_height)?;
    writeln!(out, "samples_per_pixel: {}", metrics.samples_per_pixel)?;
    writeln!(out, "max_depth: {}", metrics.max_depth)?;

    // Se escribe la semilla de la escena (si existe).
    match metrics.scene_seed {
        Some(seed) => writeln!(out, "scene_seed: {}", seed)?,
        None => writeln!(out, "scene_seed: (none)")?,
    }

    writeln!(out)?; // línea en blanco

    // ──────────────────────────────────────────────────────────────
    // Métricas de tiempo
    // ──────────────────────────────────────────────────────────────
    writeln!(out, "[timing]")?;
    writeln!(out, "render_duration_ms: {}", metrics.render_duration_ms)?;
    writeln!(out)?; // línea en blanco

    // ──────────────────────────────────────────────────────────────
    // Estadísticas específicas de NEON
    // (se escriben siempre, aunque sean 0 en el backend escalar)
    // ──────────────────────────────────────────────────────────────
    writeln!(out, "[neon_stats]")?;
    writeln!(out, "primary_rays_total: {}", metrics.primary_rays_total)?;
    writeln!(
        out,
        "primary_rays_accelerated: {}",
        metrics.primary_rays_accelerated
    )?;
    writeln!(
        out,
        "primary_rays_fallback: {}",
        metrics.primary_rays_fallback
    )?;
    writeln!(out)?; // línea en blanco

    // ──────────────────────────────────────────────────────────────
    // Estadísticas de la escena (esferas por material)
    // ──────────────────────────────────────────────────────────────
    writeln!(out, "[scene_spheres]")?;
    writeln!(out, "scene_spheres_total: {}", metrics.scene_spheres_total)?;
    writeln!(
        out,
        "scene_spheres_lambertian: {}",
        metrics.scene_spheres_lambertian
    )?;
    writeln!(
        out,
        "scene_spheres_metal: {}",
        metrics.scene_spheres_metal
    )?;
    writeln!(
        out,
        "scene_spheres_dielectric: {}",
        metrics.scene_spheres_dielectric
    )?;
    writeln!(out)?; // línea en blanco

    // ──────────────────────────────────────────────────────────────
    // Contadores lógicos del algoritmo (CoreStats)
    // ──────────────────────────────────────────────────────────────
    writeln!(out, "[core_stats]")?;
    writeln!(
        out,
        "core_rays_primary: {}",
        metrics.core_stats.rays_primary
    )?;

    writeln!(
        out,
        "core_scalar_intersection_tests: {}",
        metrics.core_stats.scalar_intersection_tests
    )?;
    writeln!(
        out,
        "core_scalar_intersection_hits: {}",
        metrics.core_stats.scalar_intersection_hits
    )?;
    writeln!(
        out,
        "core_scalar_intersection_misses: {}",
        metrics.core_stats.scalar_intersection_misses
    )?;

    writeln!(
        out,
        "core_simd_intersection_tests: {}",
        metrics.core_stats.simd_intersection_tests
    )?;
    writeln!(
        out,
        "core_simd_intersection_hits_lanes: {}",
        metrics.core_stats.simd_intersection_hits_lanes
    )?;
    writeln!(
        out,
        "core_simd_intersection_misses_lanes: {}",
        metrics.core_stats.simd_intersection_misses_lanes
    )?;

    writeln!(
        out,
        "core_lambertian_calls: {}",
        metrics.core_stats.lambertian_calls
    )?;
    writeln!(
        out,
        "core_metal_calls: {}",
        metrics.core_stats.metal_calls
    )?;
    writeln!(
        out,
        "core_dielectric_calls: {}",
        metrics.core_stats.dielectric_calls
    )?;
    writeln!(
        out,
        "core_dielectric_reflect: {}",
        metrics.core_stats.dielectric_reflect
    )?;
    writeln!(
        out,
        "core_dielectric_refract: {}",
        metrics.core_stats.dielectric_refract
    )?;

    Ok(())
}
