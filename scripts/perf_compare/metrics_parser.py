"""
metrics_parser.py

This module is responsible for parsing high-level ray tracer metrics from
metrics_*.txt files and grouping them per logical label and backend.

The expected format of metrics_*.txt is a simple key-value text file, e.g.:

    run_id: 20251130_015158_neon_403x226_spp100
    backend: neon
    render_duration_ms: 1234.56
    pps: 123456.78
    ...

All lines with the form "key: value" are considered. For each value we
attempt to parse it as a float; if parsing succeeds, it is stored in
ParsedMetrics.numeric; otherwise, it is stored as a string in
ParsedMetrics.meta.
"""

from pathlib import Path
from typing import Dict, List

from .models import ParsedMetrics, RunResult


def parse_metrics_file(path: Path) -> ParsedMetrics:
    """
    Parse a metrics_*.txt file into numeric metrics and metadata.

    Rules:
    ------
    - Lines that do not contain ':' are ignored.
    - Keys and values are stripped of surrounding whitespace.
    - 'run_id' and 'backend' are always treated as metadata (strings).
    - For other keys, we try float(value):
        * if successful  -> store in numeric
        * if it fails    -> store in meta
    """
    numeric: Dict[str, float] = {}
    meta: Dict[str, str] = {}

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        # If we cannot read the file, we treat it as empty; callers should
        # handle the fact that numeric/meta may be empty.
        print(f"[WARN] Failed to read metrics file {path}: {e}")
        return ParsedMetrics(numeric=numeric, meta=meta)

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" not in line:
            continue

        key, value_str = line.split(":", 1)
        key = key.strip()
        value_str = value_str.strip()

        if not key:
            continue

        # Always keep run_id and backend as metadata strings.
        if key in ("run_id", "backend"):
            meta[key] = value_str
            continue

        # Try to parse as float; on failure, keep as metadata string.
        try:
            value = float(value_str)
        except ValueError:
            meta[key] = value_str
        else:
            numeric[key] = value

    return ParsedMetrics(numeric=numeric, meta=meta)


def collect_highlevel_metrics(
    run_results: List[RunResult],
) -> Dict[str, Dict[str, ParsedMetrics]]:
    """
    Collect high-level metrics for each logical label and backend.

    Parameters:
    -----------
    run_results:
        Flat list of RunResult entries produced by perf_runner.run_all_with_perf.
        Each RunResult knows:
            - logical_run.label
            - backend ("scalar" / "neon")
            - metrics_file (Path to metrics_*.txt)

    Returns:
    --------
    A nested mapping with the structure:

        highlevel[label][backend] = ParsedMetrics(...)
    """
    highlevel: Dict[str, Dict[str, ParsedMetrics]] = {}

    for rr in run_results:
        label = rr.logical_run.label
        backend = rr.backend

        parsed = parse_metrics_file(rr.metrics_file)

        if label not in highlevel:
            highlevel[label] = {}

        if backend in highlevel[label]:
            # If this happens, it means we have multiple metrics files for the
            # same logical label and backend. For now we overwrite and warn.
            print(
                f"[WARN] Multiple metrics for label='{label}', backend='{backend}'. "
                f"Overwriting previous entry.",
            )

        highlevel[label][backend] = parsed

    return highlevel
