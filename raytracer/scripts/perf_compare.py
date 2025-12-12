#!/usr/bin/env python3
"""
perf_compare.py

High-level purpose:
-------------------
This script is responsible for orchestrating paired runs of the ray tracer
with two backends (currently: "scalar" and "neon"), wrapping each execution
under `perf stat`, and then comparing all collected metrics (both high-level
ray tracer metrics and microarchitectural metrics from perf).

Unlike the existing run_with_perf.py helper, this script consumes a
"meta-config" JSON that does NOT include the backend field. For each logical
run described in that meta-config, perf_compare.py will:

    - Generate two concrete configs: one with backend="scalar", one with
      backend="neon".
    - Execute the ray tracer binary twice (scalar + neon), wrapped in perf.
    - Collect and parse:
        * metrics_*.txt files produced by the ray tracer.
        * perf_*.txt files produced by this script (with derived metrics).
    - Produce comparison outputs:
        * A text file compare_<label>.txt with aligned columns.
        * Optionally, a CSV file compare_<label>.csv (if --csv is specified).

Additionally:
-------------
- If a backend field is found in the meta-config JSON, it will be ignored and
  a warning will be printed, but execution will proceed.
- Temporary files (such as per-backend JSONs and perf_tmp_*.txt) can optionally
  be deleted at the end of the run via a CLI flag (--delete-temp).

Current implementation status:
------------------------------
- Task 1: CLI definition and basic skeleton (DONE).
- Task 2: Meta-config loading, validation, label handling, and internal
          representation of logical runs (DONE).
- Next tasks will add:
    * backend-specific JSON generation,
    * perf execution,
    * metrics parsing,
    * comparison report generation.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for perf_compare.py.

    Overview of CLI design:
    -----------------------
    - Positional arguments:
        * configs: one or more "meta-config" JSON files. Each meta-config
          describes logical runs WITHOUT specifying the backend. For each
          logical run, this script will generate two concrete configs:
          backend="scalar" and backend="neon".

    - Optional arguments:
        * -b / --binary:
            Path to the compiled ray tracer binary.
            Default: ./target/release/raytracer

        * -r / --runs-dir:
            Directory where the ray tracer writes its metrics_*.txt files.
            These files typically include a run_id and various high-level
            statistics about each execution.
            Default: runs

        * --perf-dir:
            Directory where this script will write perf_<run_id>.txt files
            containing raw perf output plus derived microarchitectural metrics.
            If not provided, it defaults to the same value as --runs-dir.

        * --perf:
            Name or path of the `perf` executable.
            Default: perf

        * --delete-temp:
            If present, the script will delete temporary files that it creates
            during execution (such as backend-specific JSON configs and
            perf_tmp_*.txt files). If this flag is NOT provided, temporary
            files are preserved for debugging or inspection.

        * --csv:
            If present, the script will generate an additional CSV file
            compare_<label>.csv for each logical run, suitable for importing
            into spreadsheets or further automated analysis. If not provided,
            only the human-readable text comparison file compare_<label>.txt
            is generated.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Execute paired ray tracer runs (scalar + neon) under `perf stat`, "
            "then compare all collected metrics for each logical run defined "
            "in one or more meta-config JSON files."
        )
    )

    # Positional: one or more meta-config JSON files
    parser.add_argument(
        "configs",
        nargs="+",
        help=(
            "Paths to meta-config JSON files. Each meta-config describes "
            "logical runs WITHOUT a backend field. For each logical run, "
            "this script will generate scalar + neon concrete configs and "
            "compare their metrics."
        ),
    )

    # Path to the ray tracer binary
    parser.add_argument(
        "-b",
        "--binary",
        default="./target/release/raytracer",
        help=(
            "Path to the compiled ray tracer binary. "
            "Default: ./target/release/raytracer"
        ),
    )

    # Directory where the ray tracer writes its own metrics_*.txt files
    parser.add_argument(
        "-r",
        "--runs-dir",
        default="runs",
        help=(
            "Directory where the ray tracer writes its metrics_*.txt files. "
            "Default: runs"
        ),
    )

    # Directory where this script writes perf_<run_id>.txt files
    parser.add_argument(
        "--perf-dir",
        default=None,
        help=(
            "Directory for perf_<run_id>.txt files generated by this script. "
            "If omitted, the same directory as --runs-dir is used."
        ),
    )

    # Name/path of the `perf` executable
    parser.add_argument(
        "--perf",
        default="perf",
        help=(
            "Name or path of the `perf` executable to use. "
            "Default: perf"
        ),
    )

    # Flag to control deletion of temporary files
    parser.add_argument(
        "--delete-temp",
        action="store_true",
        help=(
            "If set, delete temporary files (backend-specific JSON configs, "
            "perf_tmp_*.txt, etc.) after all runs complete. "
            "If not set, temporary files are preserved."
        ),
    )

    # Flag to control optional CSV generation for comparisons
    parser.add_argument(
        "--csv",
        action="store_true",
        help=(
            "If set, generate an additional compare_<label>.csv file for each "
            "logical run, in addition to the human-readable text report."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Meta-config loading and label handling
# ---------------------------------------------------------------------------

def _sanitize_label(raw: str) -> str:
    """
    Sanitize a label string so that it is safe to use as part of a filename.

    The goal here is to:
        - Remove or replace characters that are problematic in file paths
          (such as '/', '\\', whitespace).
        - Preserve as much semantic information as possible.

    This function does not try to be perfect across all operating systems,
    but it is good enough for typical Linux/Unix environments.
    """
    # Replace whitespace with underscores.
    label = "_".join(raw.split())

    # Replace path separators and other obviously problematic characters.
    # More rules can be added if needed.
    forbidden_chars = ["/", "\\", ":", "*", "?", "\"", "<", ">", "|"]
    for ch in forbidden_chars:
        label = label.replace(ch, "_")

    return label


def _generate_label_from_entry(entry: Dict[str, Any], index: int) -> str:
    """
    Generate a default label for a logical run when no explicit 'label'
    field is provided in the meta-config entry.

    Heuristics:
    -----------
    We try to capture the most informative fields that are commonly present
    in ray tracer configs:
        - scene
        - image_width
        - aspect_ratio
        - samples_per_pixel
        - scene_seed

    Example pattern:
        "simple_400x225_spp10_seed3526"

    If some fields are missing, they are simply omitted. If 'scene' is not
    present, a generic base 'run<index>' is used.
    """
    scene = entry.get("scene")
    if scene is None:
        base = f"run{index}"
    else:
        base = str(scene)

    # Attempt to construct a resolution-like component (width x height).
    width = entry.get("image_width")
    aspect = entry.get("aspect_ratio")

    res_component: Optional[str] = None
    if isinstance(width, (int, float)):
        if isinstance(aspect, (int, float)) and aspect != 0:
            # Compute height as a rounded integer, derived from width/aspect.
            # This matches the usual pattern width / aspect_ratio = height.
            height = round(width / aspect)
            res_component = f"{int(width)}x{int(height)}"
        else:
            res_component = f"{int(width)}"

    parts: List[str] = [str(base)]

    if res_component is not None:
        parts.append(res_component)

    spp = entry.get("samples_per_pixel")
    if isinstance(spp, (int, float)):
        parts.append(f"spp{int(spp)}")

    seed = entry.get("scene_seed")
    if isinstance(seed, (int, float)):
        parts.append(f"seed{int(seed)}")

    # Join all pieces with underscores.
    raw_label = "_".join(parts)
    return _sanitize_label(raw_label)


def load_meta_config(path: Path) -> List[LogicalRun]:
    """
    Load a meta-config JSON file and convert it into a list of LogicalRun
    objects.

    Expected JSON structure:
    ------------------------
    {
        "run": [
            {
                "scene": "simple",
                "image_width": 400,
                "aspect_ratio": 1.7777778,
                "samples_per_pixel": 10,
                "max_depth": 50,
                "scene_seed": 3526,
                "label": "simple_400x225_spp10_seed3526",   # optional
                "backend": "scalar"                         # will be ignored
            },
            ...
        ]
    }

    Notes:
    ------
    - The 'backend' field (if present) is ignored and a warning is printed,
      because perf_compare.py *always* runs both scalar and neon backends for
      each logical run.
    - The 'label' field is used if provided; otherwise, a label is generated
      from other fields (scene, resolution, spp, seed).
    - Both 'backend' and 'label' are removed from the config_data passed to
      the ray tracer binary, so they do not interfere with its parsing.
    """
    logical_runs: List[LogicalRun] = []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        print(f"[ERROR] Failed to open meta-config file {path}: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        print(
            f"[ERROR] Invalid meta-config format in {path}: "
            f"root element must be a JSON object.",
            file=sys.stderr,
        )
        sys.exit(1)

    if "run" not in data:
        print(
            f"[ERROR] Invalid meta-config format in {path}: "
            f"missing required 'run' array.",
            file=sys.stderr,
        )
        sys.exit(1)

    runs = data["run"]
    if not isinstance(runs, list):
        print(
            f"[ERROR] Invalid meta-config format in {path}: "
            f"'run' must be a JSON array.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Track whether we have already warned about 'backend' in this file,
    # to avoid spamming the terminal if there are many runs.
    warned_backend_once = False

    # Keep track of labels to detect duplicates and avoid collisions
    # when generating filenames like compare_<label>.txt.
    used_labels: Dict[str, int] = {}

    for idx, entry in enumerate(runs):
        if not isinstance(entry, dict):
            print(
                f"[ERROR] Invalid entry in 'run' array at index {idx} in {path}: "
                f"each run must be a JSON object.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Work on a shallow copy so we can safely remove meta fields.
        entry_copy: Dict[str, Any] = dict(entry)

        # Handle 'backend' if present in the meta-config (ignore but warn).
        if "backend" in entry_copy:
            if not warned_backend_once:
                print(
                    f"[WARN] Field 'backend' found in meta-config {path}. "
                    f"It will be ignored; perf_compare.py always runs both "
                    f"scalar and neon backends.",
                    file=sys.stderr,
                )
                warned_backend_once = True
            # Remove backend from the copy so it does not propagate to the
            # ray tracer binary.
            entry_copy.pop("backend", None)

        # Determine label: use 'label' if present, otherwise generate one.
        raw_label = entry_copy.get("label")
        if isinstance(raw_label, str) and raw_label.strip():
            label = _sanitize_label(raw_label.strip())
            # Remove label from config_data; it is meta-information only.
            entry_copy.pop("label", None)
        else:
            label = _generate_label_from_entry(entry_copy, idx)

        # Ensure uniqueness of labels within the same file to avoid filename
        # collisions. If a label is repeated, append an index suffix.
        if label in used_labels:
            used_labels[label] += 1
            label_unique = f"{label}_{used_labels[label]}"
        else:
            used_labels[label] = 0
            label_unique = label

        logical_runs.append(
            LogicalRun(
                label=label_unique,
                config_data=entry_copy,
                meta_config_path=path,
                index=idx,
            )
        )

    return logical_runs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for perf_compare.py.

    Current responsibilities:
    -------------------------
    - Parse CLI arguments.
    - Resolve and normalize key paths (binary, runs-dir, perf-dir).
    - Perform basic sanity checks (e.g., binary existence).
    - Load meta-config JSON files into a list of LogicalRun objects.
    - Print a summary of:
        * The global configuration.
        * The logical runs discovered (one label per run).

    The actual logic for:
        * generating backend-specific configs,
        * running the ray tracer under perf,
        * parsing metrics and perf outputs, and
        * writing comparison reports

    will be implemented in subsequent tasks.
    """
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args()

    # Resolve important paths to absolute Paths for consistency.
    binary_path = Path(args.binary).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    perf_dir = Path(args.perf_dir).resolve() if args.perf_dir else runs_dir

    # Resolve meta-config paths as well (we will validate contents below).
    config_paths = [Path(c).resolve() for c in args.configs]

    # Basic sanity check: binary must exist.
    if not binary_path.exists():
        print(
            f"[ERROR] Ray tracer binary not found at: {binary_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load all meta-config files into LogicalRun objects
    # ------------------------------------------------------------------
    all_logical_runs: List[LogicalRun] = []
    per_file_counts: Dict[Path, int] = {}

    for cfg_path in config_paths:
        if not cfg_path.exists():
            print(
                f"[ERROR] Meta-config file not found: {cfg_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        logical_runs = load_meta_config(cfg_path)
        all_logical_runs.extend(logical_runs)
        per_file_counts[cfg_path] = len(logical_runs)

    # ------------------------------------------------------------------
    # Print configuration summary
    # ------------------------------------------------------------------
    print("[INFO] perf_compare.py configuration")
    print(f"  Binary path       : {binary_path}")
    print(f"  Runs directory    : {runs_dir}")
    print(f"  Perf directory    : {perf_dir}")
    print(f"  Perf executable   : {args.perf}")
    print(f"  Delete temp files : {args.delete_temp}")
    print(f"  Generate CSV      : {args.csv}")
    print("  Meta-config files :")
    for cfg in config_paths:
        print(f"    - {cfg}")

    print()
    print("[INFO] Loaded logical runs from meta-configs:")
    total_runs = 0
    for cfg_path in config_paths:
        count = per_file_counts.get(cfg_path, 0)
        total_runs += count
        print(f"  {cfg_path} : {count} logical run(s)")
        # Print labels for this file only (for clarity)
        for lr in all_logical_runs:
            if lr.meta_config_path == cfg_path:
                print(f"      * label: {lr.label} (index {lr.index})")

    print(f"[INFO] Total logical runs: {total_runs}")
    print()
    print("[INFO] Meta-config loading and logical run setup complete.")
    print("[INFO] Next steps (to be implemented):")
    print("       - Generate backend-specific configs (scalar/neon).")
    print("       - Run each backend under perf and collect metrics.")
    print("       - Build comparison reports (text and optional CSV).")


if __name__ == "__main__":
    main()
