// src/metrics/runlog.rs

use std::fs::{File, create_dir_all};
use std::io::{Write, Result as IoResult};
use std::path::Path;

use super::RunMetrics;

/// Escribe las métricas de una corrida en un archivo de texto.
/// Crea la carpeta padre si no existe.
pub fn write_metrics_to_file(path: &Path, metrics: &RunMetrics) -> IoResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)?;
        }
    }

    let mut file = File::create(path)?;

    writeln!(file, "run_id: {}", metrics.run_id)?;
    writeln!(file, "backend: {:?}", metrics.backend)?;
    writeln!(file, "image_width: {}", metrics.image_width)?;
    writeln!(file, "image_height: {}", metrics.image_height)?;
    writeln!(file, "samples_per_pixel: {}", metrics.samples_per_pixel)?;
    writeln!(file, "max_depth: {}", metrics.max_depth)?;
    writeln!(file, "render_duration_ms: {}", metrics.render_duration_ms)?;

    if let Some(cpu_user_ms) = metrics.cpu_user_ms {
        writeln!(file, "cpu_user_ms: {}", cpu_user_ms)?;
    } else {
        writeln!(file, "cpu_user_ms: (not measured)")?;
    }

    if let Some(cpu_system_ms) = metrics.cpu_system_ms {
        writeln!(file, "cpu_system_ms: {}", cpu_system_ms)?;
    } else {
        writeln!(file, "cpu_system_ms: (not measured)")?;
    }

    if let Some(peak_memory_bytes) = metrics.peak_memory_bytes {
        writeln!(file, "peak_memory_bytes: {}", peak_memory_bytes)?;
    } else {
        writeln!(file, "peak_memory_bytes: (not measured)")?;
    }

    Ok(())
}
