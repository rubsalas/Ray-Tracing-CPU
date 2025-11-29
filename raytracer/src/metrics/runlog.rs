// src/metrics/runlog.rs

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write, Result as IoResult};
use std::path::Path;

use crate::metrics::RunMetrics;

pub fn write_metrics_to_file(path: &Path, metrics: &RunMetrics) -> IoResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    writeln!(out, "run_id: {}", metrics.run_id)?;
    writeln!(out, "backend: {:?}", metrics.backend)?;
    writeln!(out, "image_width: {}", metrics.image_width)?;
    writeln!(out, "image_height: {}", metrics.image_height)?;
    writeln!(out, "samples_per_pixel: {}", metrics.samples_per_pixel)?;
    writeln!(out, "max_depth: {}", metrics.max_depth)?;

    // ⬇ Aquí va el scene_seed
    match metrics.scene_seed {
        Some(seed) => writeln!(out, "scene_seed: {}", seed)?,
        None       => writeln!(out, "scene_seed: (none)")?,
    }

    writeln!(out, "render_duration_ms: {}", metrics.render_duration_ms)?;
    writeln!(out, "cpu_user_ms: (not measured)")?;
    writeln!(out, "cpu_system_ms: (not measured)")?;
    writeln!(out, "peak_memory_bytes: (not measured)")?;

    // Contadores NEON, aunque sean 0 en el backend escalar.
    writeln!(out, "primary_rays_total: {}", metrics.primary_rays_total)?;
    writeln!(out, "primary_rays_accelerated: {}", metrics.primary_rays_accelerated)?;
    writeln!(out, "primary_rays_fallback: {}", metrics.primary_rays_fallback)?;

    Ok(())
}
