// src/metrics/runlog.rs

use std::fs::File;
use std::io::{Write, Result as IoResult};
use std::path::Path;

use crate::metrics::RunMetrics;

/// Escribe las métricas de una corrida en un archivo de texto.
/// Crea la carpeta padre si no existe.
pub fn write_metrics_to_file(path: &Path, m: &RunMetrics) -> IoResult<()> {
    let mut f = File::create(path)?;

    writeln!(f, "run_id: {}", m.run_id)?;
    writeln!(f, "backend: {:?}", m.backend)?;
    writeln!(f, "image_width: {}", m.image_width)?;
    writeln!(f, "image_height: {}", m.image_height)?;
    writeln!(f, "samples_per_pixel: {}", m.samples_per_pixel)?;
    writeln!(f, "max_depth: {}", m.max_depth)?;
    writeln!(f, "render_duration_ms: {}", m.render_duration_ms)?;

    writeln!(
        f,
        "cpu_user_ms: {}",
        m.cpu_user_ms
            .map(|v| v.to_string())
            .unwrap_or_else(|| "(not measured)".to_string())
    )?;
    writeln!(
        f,
        "cpu_system_ms: {}",
        m.cpu_system_ms
            .map(|v| v.to_string())
            .unwrap_or_else(|| "(not measured)".to_string())
    )?;
    writeln!(
        f,
        "peak_memory_bytes: {}",
        m.peak_memory_bytes
            .map(|v| v.to_string())
            .unwrap_or_else(|| "(not measured)".to_string())
    )?;

    // NUEVO: stats de NEON.
    // Para el backend escalar quedarán en 0, y está bien.
    writeln!(f, "primary_rays_total: {}", m.primary_rays_total)?;
    writeln!(
        f,
        "primary_rays_accelerated: {}",
        m.primary_rays_accelerated
    )?;
    writeln!(f, "primary_rays_fallback: {}", m.primary_rays_fallback)?;

    Ok(())
}
