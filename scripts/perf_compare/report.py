"""
report.py

This module generates human-readable and CSV comparison reports for each
logical label.

For each label we create:
    - A text file compare_<label>.txt with aligned columns.
    - Optionally (if requested), a compare_<label>.csv with the same data
      in machine-friendly form.

All output files are placed under a directory chosen by the caller, which
should live under scripts/perf_compare/ to keep all perf_compare artifacts
together.
"""

from pathlib import Path
from typing import Dict, List

from .models import MetricComparison


def _safe_label_for_filename(label: str) -> str:
    """
    Sanitize a label string so that it can safely be used as part of a filename.
    """
    safe = label.strip().replace(" ", "_")
    for ch in ("/", "\\", ":", ";"):
        safe = safe.replace(ch, "_")
    return safe


def _format_value(value) -> str:
    """
    Format numeric or None values as strings suitable for aligned text output.
    """
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def write_comparison_reports(
    comparisons_by_label: Dict[str, List[MetricComparison]],
    output_dir: Path,
    generate_csv: bool,
) -> None:
    """
    Write comparison reports (text + optional CSV) for each logical label.

    Parameters:
    -----------
    comparisons_by_label:
        Mapping from label to list of MetricComparison objects.

    output_dir:
        Directory where compare_<label>.txt and compare_<label>.csv files
        should be written. The directory is created if it does not exist.

    generate_csv:
        If True, generate CSV files in addition to the text reports.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[ERROR] Failed to create reports directory {output_dir}: {e}")
        return

    for label, comparisons in comparisons_by_label.items():
        if not comparisons:
            continue

        safe_label = _safe_label_for_filename(label)

        txt_path = output_dir / f"compare_{safe_label}.txt"
        csv_path = output_dir / f"compare_{safe_label}.csv"

        # Determine column widths for pretty alignment
        metric_col_w = max(24, max(len(c.metric_name) for c in comparisons))
        value_col_w = 18

        header = (
            f"{'Metric':<{metric_col_w}} "
            f"{'Scalar':>{value_col_w}} "
            f"{'NEON':>{value_col_w}} "
            f"{'Diff (NEON - scalar)':>{value_col_w}} "
            f"{'Ratio (NEON / scalar)':>{value_col_w}} "
            f"{'Better':>8}"
        )

        separator = "-" * len(header)

        # Write text report
        try:
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(f"# Comparison report for label: {label}\n")
                f.write(header + "\n")
                f.write(separator + "\n")

                for comp in comparisons:
                    scalar_str = _format_value(comp.scalar_value)
                    neon_str = _format_value(comp.neon_value)
                    diff_str = _format_value(comp.diff_neon_minus_scalar)
                    ratio_str = _format_value(comp.ratio_neon_over_scalar)

                    line = (
                        f"{comp.metric_name:<{metric_col_w}} "
                        f"{scalar_str:>{value_col_w}} "
                        f"{neon_str:>{value_col_w}} "
                        f"{diff_str:>{value_col_w}} "
                        f"{ratio_str:>{value_col_w}} "
                        f"{comp.better:>8}"
                    )
                    f.write(line + "\n")
        except OSError as e:
            print(f"[ERROR] Failed to write text report {txt_path}: {e}")
            continue

        # Optionally write CSV report
        if generate_csv:
            try:
                with csv_path.open("w", encoding="utf-8") as f:
                    f.write(
                        "metric_name,scalar_value,neon_value,"
                        "diff_neon_minus_scalar,ratio_neon_over_scalar,better\n"
                    )
                    for comp in comparisons:
                        s_val = "" if comp.scalar_value is None else f"{comp.scalar_value:.6f}"
                        n_val = "" if comp.neon_value is None else f"{comp.neon_value:.6f}"
                        d_val = (
                            "" if comp.diff_neon_minus_scalar is None
                            else f"{comp.diff_neon_minus_scalar:.6f}"
                        )
                        r_val = (
                            "" if comp.ratio_neon_over_scalar is None
                            else f"{comp.ratio_neon_over_scalar:.6f}"
                        )
                        f.write(
                            f"{comp.metric_name},{s_val},{n_val},"
                            f"{d_val},{r_val},{comp.better}\n"
                        )
            except OSError as e:
                print(f"[ERROR] Failed to write CSV report {csv_path}: {e}")
                continue

        print(f"[INFO] Wrote comparison report(s) for label='{label}': {txt_path}")
        if generate_csv:
            print(f"[INFO] Wrote CSV comparison report for label='{label}': {csv_path}")
