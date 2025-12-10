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

This file currently only defines the command-line interface (CLI) and the
overall skeleton. The detailed implementation for each step will be added
incrementally in subsequent tasks.

Complete example command:
python3 scripts/perf_compare.py \
    -b ./target/release/raytracer \
    -r runs \
    --perf-dir perf_runs \
    --perf perf \
    --delete-temp \
    --csv \
    config/exp_simple.json config/exp_many_spheres.json

"""

import argparse
import sys
from pathlib import Path
from typing import List


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


def main(argv: List[str] | None = None) -> None:
    """
    Main entry point for perf_compare.py.

    Current responsibilities:
    -------------------------
    - Parse CLI arguments.
    - Resolve and normalize key paths (binary, runs-dir, perf-dir).
    - Perform basic sanity checks (e.g., binary existence).
    - Print a short summary of the configuration being used.

    The actual logic for:
        * reading meta-config JSONs,
        * generating backend-specific configs,
        * running the ray tracer under perf,
        * parsing metrics and perf outputs, and
        * writing comparison reports

    will be implemented in subsequent tasks. For now, this function acts as a
    verified skeleton to ensure the CLI design is correct and all paths are
    handled consistently.
    """
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args()

    # Resolve important paths to absolute Paths for consistency.
    binary_path = Path(args.binary).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    perf_dir = Path(args.perf_dir).resolve() if args.perf_dir else runs_dir

    # Resolve meta-config paths as well (but do not validate contents yet).
    config_paths = [Path(c).resolve() for c in args.configs]

    # Basic sanity check: binary must exist.
    if not binary_path.exists():
        print(
            f"[ERROR] Ray tracer binary not found at: {binary_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Print a small summary to confirm the parsed configuration.
    # This is useful for debugging and for ensuring that the CLI behaves
    # exactly as expected before we implement the heavy logic.
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

    # TODO (next tasks):
    #  - For each meta-config, load and validate JSON structure.
    #  - For each logical run in the meta-config, generate scalar + neon configs.
    #  - Execute the ray tracer under perf for each backend.
    #  - Collect metrics and perf outputs, then build comparison reports.
    #  - Optionally delete temporary files if args.delete_temp is True.
    #  - Optionally generate CSV comparison files if args.csv is True.

    # For now, we simply exit after printing the summary so we can
    # incrementally build and test the rest of the pipeline.
    print("[INFO] Initialization complete. Further logic will be added in next tasks.")


if __name__ == "__main__":
    main()
