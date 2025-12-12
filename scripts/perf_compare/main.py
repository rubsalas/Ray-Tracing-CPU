"""
Main orchestrator for perf_compare.

This module wires together:
    - CLI parsing (cli.parse_args),
    - meta-config loading (meta_config.load_meta_config),
    - backend-specific config generation (backend_configs.generate_backend_configs),
    - perf execution and metrics mapping (perf_runner.run_all_with_perf),

and prints a detailed summary.

Later tasks will extend this main entry point to:
    - Parse high-level ray tracer metrics,
    - Compare scalar vs neon per logical run,
    - Emit comparison reports (text + optional CSV),
    - Write all outputs under scripts/perf_compare/.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

from .cli import parse_args
from .meta_config import load_meta_config
from .backend_configs import generate_backend_configs
from .perf_runner import run_all_with_perf
from .metrics_parser import collect_highlevel_metrics
from .compare import build_highlevel_comparisons
from .report import write_comparison_reports
from .models import LogicalRun, BackendRun, RunResult



def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for perf_compare.

    Current responsibilities:
    -------------------------
    - Parse CLI arguments.
    - Resolve and normalize key paths (binary, runs-dir, perf-dir, rust-config-dir).
    - Perform basic sanity checks (e.g., binary existence).
    - Load meta-config JSON files into a list of LogicalRun objects.
    - Generate backend-specific JSON configs (scalar + neon) under raytracer/config/.
    - Run perf for each backend config, collect metrics and perf outputs, and
      build a mapping of results per logical label and backend.
    - Optionally delete temporary files (--delete-temp).
    - Print a summary of:
        * Global configuration.
        * Logical runs discovered (labels, indices).
        * Backend-specific configs generated.
        * Perf results (run_id per label/backend).
    """
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args()

    # All these paths are resolved relative to the current working directory,
    # which is expected to be the *repository root* that contains both
    # 'raytracer/' and 'scripts/'.
    binary_path = Path(args.binary).resolve()
    runs_dir = Path(args.runs_dir).resolve()

    # Perf directory:
    # By default we keep all perf-related outputs under scripts/perf_compare/perf_runs
    # so that everything produced by perf_compare stays inside scripts/perf_compare/.
    if args.perf_dir:
        perf_dir = Path(args.perf_dir).resolve()
    else:
        perf_dir = Path("scripts/perf_compare/perf_runs").resolve()

    config_paths = [Path(c).resolve() for c in args.configs]

    # Rust config directory (where the ray tracer expects its JSON configs).
    # Layout:
    #   <root>/
    #     raytracer/
    #       config/
    rust_config_dir = Path("raytracer/config").resolve()

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
    # Generate backend-specific JSON configs (scalar + neon) under raytracer/config/
    # ------------------------------------------------------------------
    backend_runs: List[BackendRun]
    temp_config_files: List[Path]
    backend_runs, temp_config_files = generate_backend_configs(
        all_logical_runs,
        rust_config_dir,
    )

    # ------------------------------------------------------------------
    # Run perf for each backend config and collect results
    # ------------------------------------------------------------------
    run_results_list: List[RunResult]
    run_results_map: Dict[str, Dict[str, RunResult]]
    perf_temp_files: List[Path]

    run_results_list, run_results_map, perf_temp_files = run_all_with_perf(
        perf_cmd=args.perf,
        binary_path=binary_path,
        runs_dir=runs_dir,
        perf_dir=perf_dir,
        backend_runs=backend_runs,
    )

    # ------------------------------------------------------------------
    # Optionally delete temporary files (--delete-temp)
    # ------------------------------------------------------------------
    if args.delete_temp:
        print("[INFO] --delete-temp: removing temporary files...")
        # Backend-specific JSON configs in raytracer/config/
        for p in temp_config_files:
            try:
                p.unlink()
                print(f"[INFO] Deleted temp config file: {p}")
            except OSError as e:
                print(f"[WARN] Failed to delete temp config file {p}: {e}", file=sys.stderr)
        # perf_tmp_*.txt files in perf_dir
        for p in perf_temp_files:
            try:
                p.unlink()
                print(f"[INFO] Deleted temp perf file: {p}")
            except OSError as e:
                print(f"[WARN] Failed to delete temp perf file {p}: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Parse high-level metrics and build scalar vs neon comparisons
    # ------------------------------------------------------------------
    comparisons_by_label = {}

    if run_results_list:
        print()
        print("[INFO] Parsing high-level metrics from metrics_*.txt...")
        highlevel = collect_highlevel_metrics(run_results_list)

        print("[INFO] Building scalar vs NEON comparisons for high-level metrics...")
        comparisons_by_label = build_highlevel_comparisons(highlevel)

        reports_dir = Path("scripts/perf_compare/reports").resolve()
        write_comparison_reports(
            comparisons_by_label=comparisons_by_label,
            output_dir=reports_dir,
            generate_csv=args.csv,
        )
    else:
        print()
        print("[WARN] No RunResult entries recorded; skipping metric comparison and report generation.")

    # ------------------------------------------------------------------
    # Print configuration and results summary
    # ------------------------------------------------------------------
    print()
    print("[INFO] perf_compare configuration")
    print(f"  Binary path       : {binary_path}")
    print(f"  Runs directory    : {runs_dir}")
    print(f"  Perf directory    : {perf_dir}")
    print(f"  Rust config dir   : {rust_config_dir}")
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
        for lr in all_logical_runs:
            if lr.meta_config_path == cfg_path:
                print(f"      * label: {lr.label} (index {lr.index})")
    print(f"[INFO] Total logical runs: {total_runs}")

    print()
    print("[INFO] Generated backend-specific configs (stored in raytracer/config/):")
    for br in backend_runs:
        print(
            f"  label={br.logical_run.label:<40} "
            f"backend={br.backend:<6} "
            f"config={br.config_path}"
        )

    print()
    print("[INFO] Perf execution results (per label/backend):")
    if not run_results_map:
        print("  [WARN] No RunResult entries recorded. Check warnings above.")
    else:
        for label, by_backend in run_results_map.items():
            print(f"  label={label}")
            for backend, rr in by_backend.items():
                print(
                    f"    backend={backend:<6} "
                    f"run_id={rr.run_id} "
                    f"metrics={rr.metrics_file.name} "
                    f"perf={rr.perf_file.name}"
                )

    if comparisons_by_label:
        print()
        print("[INFO] High-level metric comparison reports generated under:")
        print(f"       {Path('scripts/perf_compare/reports').resolve()}")
    else:
        print()
        print("[INFO] No high-level comparison reports were generated.")

    print()
    print("[INFO] perf_compare pipeline complete.")


if __name__ == "__main__":
    main()
