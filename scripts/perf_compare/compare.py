"""
compare.py

This module builds scalar vs neon comparisons for high-level metrics parsed
from metrics_*.txt files.

Given a mapping:

    highlevel[label][backend] = ParsedMetrics

it produces, for each label, a list of MetricComparison objects that capture:

    - scalar value
    - neon value
    - neon - scalar
    - neon / scalar
    - which backend looks "better" under simple heuristics
"""

from typing import Dict, List, Optional

from .models import ParsedMetrics, MetricComparison


def _classify_metric_trend(metric_name: str) -> str:
    """
    Classify how we expect a given metric to behave in terms of "better":

        - 'lower_is_better'
        - 'higher_is_better'
        - 'structural'   -> configuration / scene description; we don't judge
        - 'counter_only' -> internal counters; we don't judge
        - 'unknown'      -> no clear trend; we don't judge

    We apply explicit rules for well-known performance metrics and use
    conservative heuristics for the rest.
    """
    name = metric_name.lower()

    # --- 0) Explicit allowlist for known performance metrics ---
    # Here we explicitly say "we will judge these":
    if name in ("render_duration_ms", "render_time_ms"):
        return "lower_is_better"

    if name in (
        "core_scalar_intersection_tests",
        "core_scalar_intersection_misses",
    ):
        # Fewer scalar intersection tests / misses is better:
        # less scalar work for the same rays.
        return "lower_is_better"

    # (cuando integremos perf, aquí podemos agregar: 'ipc', 'pps', etc.)

    # --- 1) Structural / config-like metrics: we don't judge "better" ---
    if (
        name.startswith("image_")
        or name.startswith("scene_")
        or name in ("samples_per_pixel", "max_depth", "scene_seed")
    ):
        return "structural"

    # --- 2) Internal counters where "more" or "less" is not clearly better ---
    # SIMD-specific internal counters
    if name.startswith("core_simd_") or name.startswith("primary_rays_"):
        return "counter_only"

    # Generic call counters (e.g. core_dielectric_calls)
    if name.endswith("_calls"):
        return "counter_only"

    # --- 3) Generic time/duration heuristics (fallback) ---
    if any(tok in name for tok in ["duration", "latency"]):
        return "lower_is_better"
    if name.endswith("_ms") or name.endswith("_s"):
        return "lower_is_better"

    # --- 4) Generic throughput / rate heuristics (fallback) ---
    if any(tok in name for tok in ["pps", "per_sec", "throughput", "ips"]):
        return "higher_is_better"

    if name == "ipc":
        return "higher_is_better"

    # --- 5) Error / miss rates, MPKI: lower is better ---
    if any(tok in name for tok in ["_miss_rate", "error_rate"]):
        return "lower_is_better"

    if name.endswith("mpki"):
        return "lower_is_better"

    # If the metric does not clearly fall into any known pattern,
    # we treat it conservatively as unknown.
    return "unknown"



def _decide_better(
    metric_name: str,
    scalar_value: Optional[float],
    neon_value: Optional[float],
) -> str:
    """
    Decide which backend looks better based on the metric trend classification
    and the scalar/neon values.

    Returns:
        'scalar', 'neon', 'tie', or 'n/a'.
    """
    if scalar_value is None or neon_value is None:
        return "n/a"

    if scalar_value == neon_value:
        return "tie"

    trend = _classify_metric_trend(metric_name)

    if trend in ("structural", "counter_only", "unknown"):
        # Do not try to assign "better" semantics here; we only show the values.
        return "n/a"

    if trend == "higher_is_better":
        return "neon" if neon_value > scalar_value else "scalar"

    if trend == "lower_is_better":
        return "neon" if neon_value < scalar_value else "scalar"

    return "n/a"



def build_highlevel_comparisons(
    highlevel: Dict[str, Dict[str, ParsedMetrics]],
) -> Dict[str, List[MetricComparison]]:
    """
    Build scalar vs neon metric comparisons per logical label.

    Parameters:
    -----------
    highlevel:
        Mapping of:
            highlevel[label][backend] = ParsedMetrics

    Returns:
    --------
    comparisons_by_label:
        Dictionary:
            comparisons_by_label[label] = [MetricComparison, ...]
    """
    comparisons_by_label: Dict[str, List[MetricComparison]] = {}

    for label, by_backend in highlevel.items():
        parsed_scalar = by_backend.get("scalar")
        parsed_neon = by_backend.get("neon")

        if not parsed_scalar and not parsed_neon:
            # Nothing to compare for this label
            continue

        scalar_numeric = parsed_scalar.numeric if parsed_scalar else {}
        neon_numeric = parsed_neon.numeric if parsed_neon else {}

        # Use the union of metric names to "compare all metrics we have"
        all_metric_names = sorted(set(scalar_numeric.keys()) | set(neon_numeric.keys()))

        label_comparisons: List[MetricComparison] = []

        for metric_name in all_metric_names:
            s_val = scalar_numeric.get(metric_name)
            n_val = neon_numeric.get(metric_name)

            diff: Optional[float] = None
            ratio: Optional[float] = None

            if s_val is not None and n_val is not None:
                diff = n_val - s_val
                if s_val != 0.0:
                    ratio = n_val / s_val

            better = _decide_better(metric_name, s_val, n_val)

            label_comparisons.append(
                MetricComparison(
                    metric_name=metric_name,
                    scalar_value=s_val,
                    neon_value=n_val,
                    diff_neon_minus_scalar=diff,
                    ratio_neon_over_scalar=ratio,
                    better=better,
                )
            )

        comparisons_by_label[label] = label_comparisons

    return comparisons_by_label
