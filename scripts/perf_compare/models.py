from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Backends that will be compared for each logical run. Keeping this here
# makes it easy to extend (e.g., adding "simd", "gpu" backends) without
# touching the rest of the code.
BACKENDS: List[str] = ["scalar", "neon"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LogicalRun:
    """
    Represents a single *logical* run as defined in a meta-config JSON.

    A logical run:
    --------------
    - Comes from a specific meta-config file.
    - Corresponds to one entry within the "run" array in that file.
    - Does NOT yet include any backend information (scalar/neon).
      Backends will be applied later when generating concrete configs.

    Fields:
    -------
    label:
        Human-readable identifier for this run. Also used to build filenames
        such as compare_<label>.txt and compare_<label>.csv.

    config_data:
        A dictionary containing the configuration fields that should be passed
        to the ray tracer binary. This dict is derived from the original JSON
        entry, but fields that are purely meta (such as "label" or "backend")
        are removed to avoid confusing the binary.

    meta_config_path:
        Path to the meta-config JSON file from which this logical run was
        loaded. Useful for debugging and logging.

    index:
        Zero-based index of this entry within the "run" array inside
        meta_config_path. Also useful for debugging and for constructing
        fallback labels if necessary.
    """
    label: str
    config_data: Dict[str, Any]
    meta_config_path: Path
    index: int


@dataclass
class BackendRun:
    """
    Represents a concrete run configuration for a specific backend.

    While LogicalRun is backend-agnostic, BackendRun ties together:
        - The original LogicalRun (for context, label, meta-config origin).
        - The chosen backend (e.g., "scalar" or "neon").
        - The path to the backend-specific JSON config file that will be
          passed directly to the ray tracer binary.

    This abstraction is useful because:
        - The execution layer (perf + ray tracer) can simply iterate over
          BackendRun instances.
        - Later, we can attach additional per-backend metadata (run_id,
          metrics paths, perf paths, etc.) if needed.
    """
    logical_run: LogicalRun
    backend: str
    config_path: Path


@dataclass
class RunResult:
    """
    Represents the outcome of executing a BackendRun under perf.

    Fields:
    -------
    logical_run:
        The original logical run from which this backend execution was derived.

    backend:
        The backend used for this run ("scalar", "neon", etc.).

    run_id:
        The run_id parsed from the ray tracer metrics file. This is the key
        that ties together:
            - metrics_*.txt (ray tracer metrics)
            - perf_<run_id>.txt (perf metrics + derived metrics)

    metrics_file:
        Path to the metrics_*.txt file produced by the ray tracer.

    perf_file:
        Path to the perf_<run_id>.txt file produced by this tool, which
        contains raw perf output plus a block of derived microarchitectural
        metrics.
    """
    logical_run: LogicalRun
    backend: str
    run_id: str
    metrics_file: Path
    perf_file: Path

@dataclass
class ParsedMetrics:
    """
    Parsed content of a metrics_*.txt file.

    numeric:
        Dictionary of numeric metrics parsed as floats. Keys are metric names
        (e.g. 'render_duration_ms', 'pps', 'rays_traced'), values are floats.

    meta:
        Dictionary of non-numeric or descriptive fields (e.g. 'run_id',
        'backend', 'scene_name', etc.). These are kept as strings and are not
        used directly in numeric comparisons, but may be useful for debugging
        or future reporting.
    """
    numeric: Dict[str, float]
    meta: Dict[str, str]


@dataclass
class MetricComparison:
    """
    Comparison of a single numeric metric between scalar and neon backends
    for a given logical run label.

    Fields:
    -------
    metric_name:
        Name of the metric (e.g. 'render_duration_ms', 'pps').

    scalar_value:
        Numeric value from the scalar run, if available. None if the metric
        was missing in scalar.

    neon_value:
        Numeric value from the neon run, if available. None if the metric
        was missing in neon.

    diff_neon_minus_scalar:
        Difference neon - scalar, if both values are available. None otherwise.

    ratio_neon_over_scalar:
        Ratio neon / scalar, if scalar_value is non-zero and both values are
        available. None otherwise.

    better:
        Which backend looks "better" according to simple heuristics:
            - 'scalar'
            - 'neon'
            - 'tie'
            - 'n/a' (not applicable or unknown trend)
    """
    metric_name: str
    scalar_value: Optional[float]
    neon_value: Optional[float]
    diff_neon_minus_scalar: Optional[float]
    ratio_neon_over_scalar: Optional[float]
    better: str
