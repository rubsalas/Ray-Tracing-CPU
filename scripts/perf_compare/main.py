"""
Main orchestrator for perf_compare.

This module wires together:
    - CLI parsing (cli.parse_args),
    - meta-config loading (meta_config.load_meta_config),
    - backend-specific config generation (backend_configs.generate_backend_configs),

and prints a detailed summary.

Later tasks will extend this main entry point to:
    - Run each backend under perf,
    - Parse high-level and microarchitectural metrics,
    - Emit comparison reports (text + optional CSV),
    - Optionally delete temporary files (--delete-temp).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

from .cli import parse_args
from .meta_config import load_meta_config
from .backend_configs import generate_backend_configs
from .models import LogicalRun, BackendRun


def main(argv: Optional[List[str]] = None) -> None:
    """
    Main entry point for perf_compare.

    Current responsibilities:
    -------------------------
    - Parse CLI arguments.
    - Resolve and normalize key paths (binary, runs-dir, perf-dir).
    - Perform basic sanity checks (e.g., binary existence).
    - Load meta-config JSON files into a list of LogicalRun objects.
    - Generate backend-specific JSON configs (scalar + neon) for each
      LogicalRun and keep track of them as temporary files.
    - Print a summary of:
        * The global configuration.
        * The logical runs discovered (one label per run).
        * The backend-specific config files that were generated.
    """
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args()

    # All these paths are resolved relative to the current working directory,
    # which is expected to be the repository root that contains both
    # 'raytracer/' and 'scripts/'.
    binary_path = Path(args.binary).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    perf_dir = Path(args.perf_dir).resolve() if args.perf_dir else runs_dir
    config_paths = [Path(c).resolve() for c in args.configs]

    # 🔹 NUEVO: directorio de configs del raytracer Rust
    # Asumimos layout:
    #   <root>/
    #     raytracer/
    #       config/
    #     scripts/
    #
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
    temp_files: List[Path]
    backend_runs, temp_files = generate_backend_configs(
        all_logical_runs,
        rust_config_dir,
    )

    # ------------------------------------------------------------------
    # Print configuration summary
    # ------------------------------------------------------------------
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

    print("[INFO] Generated backend-specific configs:")
    for br in backend_runs:
        print(
            f"  label={br.logical_run.label:<40} "
            f"backend={br.backend:<6} "
            f"config={br.config_path}"
        )

    print()
    print("[INFO] Backend config generation complete.")
    print("[INFO] Next steps (to be implemented):")
    print("       - Run each backend under perf and collect metrics.")
    print("       - Build comparison reports (text and optional CSV).")
    print("       - Optionally delete temporary files if --delete-temp is set.")


if __name__ == "__main__":
    main()
