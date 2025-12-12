"""
perf_runner.py

This module is responsible for:
    - Running the ray tracer under `perf stat` for each BackendRun.
    - Tracking which metrics_*.txt files are created by each execution.
    - Parsing perf output and computing derived microarchitectural metrics.
    - Writing perf_<run_id>.txt files under a dedicated perf directory.
    - Returning a structured mapping of results per logical label and backend.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import BackendRun, RunResult


# ---------------------------------------------------------------------------
# Perf events
# ---------------------------------------------------------------------------

# Events requested from `perf stat`. This mirrors the original run_with_perf.py.
PERF_EVENTS: List[str] = [
    # basic
    "task-clock",
    "context-switches",
    "cpu-migrations",
    "page-faults",
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    # caches
    "l1d_cache",
    "l1d_cache_refill",
    "l1i_cache",
    "l1i_cache_refill",
    "l2d_cache",
    "l2d_cache_refill",
    "l3d_cache",
    "l3d_cache_refill",
    "ll_cache_rd",
    "ll_cache_miss_rd",
]


# ---------------------------------------------------------------------------
# Helpers for ray tracer metrics files
# ---------------------------------------------------------------------------

def list_metric_files(runs_dir: Path) -> Set[Path]:
    """
    Return the set of metric files currently present in runs_dir.

    We assume the ray tracer writes its metrics as *.txt files under the
    runs_dir (for example, metrics_<run_id>.txt). We do not enforce a
    specific naming pattern here, only the extension, because the original
    helper behaved the same way and perf outputs are kept elsewhere.
    """
    if not runs_dir.exists():
        return set()
    return set(runs_dir.glob("*.txt"))


def extract_run_id(metrics_file: Path) -> str:
    """
    Extract the value of 'run_id' from a metrics file.

    The function scans the file for a line starting with 'run_id:' and
    returns the text after the first colon.

    Example line:
        run_id: 20251130_015158_neon_403x226_spp100

    If no such line is found, a ValueError is raised.
    """
    try:
        with metrics_file.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("run_id:"):
                    return stripped.split(":", 1)[1].strip()
    except OSError as e:
        raise ValueError(f"Failed to read metrics file {metrics_file}: {e}") from e

    raise ValueError(f"No 'run_id:' line found in metrics file {metrics_file}")


# ---------------------------------------------------------------------------
# Perf output parsing and derived metrics
# ---------------------------------------------------------------------------

def parse_perf_metrics(perf_text: str) -> Dict[str, float]:
    """
    Parse the output of `perf stat` and return a dict metric_name -> value.

    Supported patterns:
    -------------------
    - Lines like:
          11764165449      cycles
          9273841591       l1d_cache

    - task-clock in msec:
          4920.93 msec task-clock

    - final time lines:
          4.923556772 seconds time elapsed
          4.911326000 seconds user
          0.009996000 seconds sys
    """
    metrics: Dict[str, float] = {}

    for line in perf_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()

        # Elapsed time line:
        #   4.923556772 seconds time elapsed
        if len(parts) >= 4 and parts[1] == "seconds" and parts[2] == "time" and parts[3] == "elapsed":
            try:
                metrics["time_elapsed"] = float(parts[0])
            except ValueError:
                pass
            continue

        # User/sys time lines:
        #   4.911326000 seconds user
        #   0.009996000 seconds sys
        if len(parts) >= 3 and parts[1] == "seconds" and parts[2] in ("user", "sys"):
            key = f"time_{parts[2]}"
            try:
                metrics[key] = float(parts[0])
            except ValueError:
                pass
            continue

        first = parts[0]
        if not first:
            continue
        if not (first[0].isdigit() or first[0] == "."):
            continue

        # Remove thousands separators if any.
        value_str = first.replace(",", "")
        try:
            value = float(value_str)
        except ValueError:
            continue

        metric_name: Optional[str] = None

        # task-clock value with unit "msec":
        #   4920.93 msec task-clock
        if len(parts) >= 3 and parts[1] == "msec":
            metric_name = parts[2]
            if metric_name == "task-clock":
                metrics["task_clock_msec"] = value
                metrics["task_clock_sec"] = value / 1000.0
            else:
                metrics[metric_name] = value
        elif len(parts) >= 2:
            metric_name = parts[1]
            metrics[metric_name] = value

    return metrics


def compute_derived_metrics(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Compute derived microarchitectural metrics from raw perf counters.

    The derived metrics can include:
        - ipc
        - branch_miss_rate
        - l1d_miss_rate, l1i_miss_rate, l2d_miss_rate, llc_miss_rate
        - l1d_mpki, l2d_mpki, llc_mpki
        - inst_per_sec
    """
    def get(name: str) -> Optional[float]:
        return raw.get(name)

    def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
        if num is None or den is None or den == 0:
            return None
        return num / den

    derived: Dict[str, float] = {}

    cycles = get("cycles")
    instructions = get("instructions")
    branches = get("branches")
    branch_misses = get("branch-misses")
    time_elapsed = get("time_elapsed")

    l1d = get("l1d_cache")
    l1d_refill = get("l1d_cache_refill")
    l1i = get("l1i_cache")
    l1i_refill = get("l1i_cache_refill")
    l2d = get("l2d_cache")
    l2d_refill = get("l2d_cache_refill")
    ll_rd = get("ll_cache_rd")
    ll_miss_rd = get("ll_cache_miss_rd")

    # IPC
    ipc = safe_div(instructions, cycles)
    if ipc is not None:
        derived["ipc"] = ipc

    # Branch miss rate
    bmr = safe_div(branch_misses, branches)
    if bmr is not None:
        derived["branch_miss_rate"] = bmr

    # Cache miss rates
    l1d_mr = safe_div(l1d_refill, l1d)
    if l1d_mr is not None:
        derived["l1d_miss_rate"] = l1d_mr

    l1i_mr = safe_div(l1i_refill, l1i)
    if l1i_mr is not None:
        derived["l1i_miss_rate"] = l1i_mr

    l2d_mr = safe_div(l2d_refill, l2d)
    if l2d_mr is not None:
        derived["l2d_miss_rate"] = l2d_mr

    llc_mr = safe_div(ll_miss_rd, ll_rd)
    if llc_mr is not None:
        derived["llc_miss_rate"] = llc_mr

    # MPKI (misses per kilo instructions)
    if instructions and instructions != 0:
        if l1d_refill is not None:
            derived["l1d_mpki"] = 1000.0 * l1d_refill / instructions
        if l2d_refill is not None:
            derived["l2d_mpki"] = 1000.0 * l2d_refill / instructions
        if ll_miss_rd is not None:
            derived["llc_mpki"] = 1000.0 * ll_miss_rd / instructions

    # Instruction throughput
    ips = safe_div(instructions, time_elapsed)
    if ips is not None:
        derived["inst_per_sec"] = ips

    return derived


def format_derived_metrics(derived: Dict[str, float]) -> str:
    """
    Format the derived metrics into a human-readable block of text, grouped
    by categories (branch, L1, L2, LLC, IPC/throughput).
    """
    lines: List[str] = []
    lines.append("# ==== Derived metrics (computed by perf_runner.py) ====")

    printed: Set[str] = set()

    def add_rate(name: str) -> None:
        if name in derived:
            val = derived[name]
            lines.append(f"{name}: {val:.6f} ({val*100:.3f}%)")
            printed.add(name)

    def add_mpki(name: str) -> None:
        if name in derived:
            val = derived[name]
            lines.append(f"{name}: {val:.4f} misses / kilo-instructions")
            printed.add(name)

    # 1) Branch
    add_rate("branch_miss_rate")

    # 2) L1
    add_rate("l1d_miss_rate")
    add_mpki("l1d_mpki")
    add_rate("l1i_miss_rate")

    # 3) L2
    add_rate("l2d_miss_rate")
    add_mpki("l2d_mpki")

    # 4) LLC
    add_rate("llc_miss_rate")
    add_mpki("llc_mpki")

    # 5) IPC / throughput
    if "ipc" in derived:
        val = derived["ipc"]
        lines.append(f"ipc: {val:.4f} instructions / cycle")
        printed.add("ipc")

    if "inst_per_sec" in derived:
        val = derived["inst_per_sec"]
        lines.append(f"inst_per_sec: {val:.2f} instructions / second")
        printed.add("inst_per_sec")

    # 6) Any remaining derived metrics not already printed
    remaining = sorted(k for k in derived.keys() if k not in printed)
    for key in remaining:
        val = derived[key]
        if "rate" in key:
            lines.append(f"{key}: {val:.6f} ({val*100:.3f}%)")
        elif "mpki" in key:
            lines.append(f"{key}: {val:.4f} misses / kilo-instructions")
        elif key == "ipc":
            lines.append(f"{key}: {val:.4f} instructions / cycle")
        elif key == "inst_per_sec":
            lines.append(f"{key}: {val:.2f} instructions / second")
        else:
            lines.append(f"{key}: {val:.6f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core perf execution logic
# ---------------------------------------------------------------------------

def run_all_with_perf(
    perf_cmd: str,
    binary_path: Path,
    runs_dir: Path,
    perf_dir: Path,
    backend_runs: List[BackendRun],
) -> Tuple[List[RunResult], Dict[str, Dict[str, RunResult]], List[Path]]:
    """
    Execute the ray tracer under perf for each BackendRun.

    For each BackendRun:
        - Take a snapshot of metrics_*.txt files in runs_dir ("before").
        - Run `perf stat` around `<binary> <backend_config.json>`, writing
          its output to a temporary perf_tmp_*.txt file under perf_dir.
        - Read the temporary perf output, parse raw metrics, compute derived
          metrics, and append them to create perf_<run_id>.txt.
        - Take another snapshot of metrics_*.txt ("after") and detect which
          files are new.
        - For each new metrics file, extract run_id and create a RunResult,
          mapping it by logical label and backend.

    Returns:
    --------
    results_list:
        Flat list of RunResult objects (one per run_id detected).

    results_map:
        Nested mapping:
            results_map[logical_label][backend] = RunResult

        This is useful for subsequent comparison steps.

    perf_temp_files:
        List of temporary perf_tmp_*.txt files created during the process.
        These can be deleted if the user passed --delete-temp.
    """
    results_list: List[RunResult] = []
    results_map: Dict[str, Dict[str, RunResult]] = {}
    perf_temp_files: List[Path] = []

    # Ensure directories exist
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[ERROR] Failed to create or access runs directory {runs_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        perf_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[ERROR] Failed to create or access perf directory {perf_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    events_arg = ",".join(PERF_EVENTS)

    for br in backend_runs:
        # Snapshot of metrics files before the run
        before_files = list_metric_files(runs_dir)

        # Build a unique temporary perf output file name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        perf_tmp = perf_dir / f"perf_tmp_{timestamp}_{br.logical_run.label}_{br.backend}.txt"

        cmd: List[str] = [
            perf_cmd,
            "stat",
            "-e",
            events_arg,
            "-o",
            str(perf_tmp),
            "--",
            str(binary_path),
            str(br.config_path),
        ]

        print(f"[INFO] Running perf for label='{br.logical_run.label}', backend='{br.backend}'")
        print(f"[INFO] Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"[ERROR] perf or the ray tracer binary failed for "
                f"label='{br.logical_run.label}', backend='{br.backend}': {e}",
                file=sys.stderr,
            )
            # Keep the temporary file for inspection; do not abort all runs.
            perf_temp_files.append(perf_tmp)
            continue

        perf_temp_files.append(perf_tmp)

        # Read perf output and compute derived metrics
        try:
            perf_text = perf_tmp.read_text(encoding="utf-8")
        except OSError as e:
            print(f"[ERROR] Failed to read temporary perf file {perf_tmp}: {e}", file=sys.stderr)
            continue

        raw_metrics = parse_perf_metrics(perf_text)
        derived = compute_derived_metrics(raw_metrics)
        derived_block = format_derived_metrics(derived) if derived else ""

        combined_perf_text = perf_text
        if derived_block:
            combined_perf_text = perf_text.rstrip() + "\n\n" + derived_block + "\n"

        # Snapshot after the run
        after_files = list_metric_files(runs_dir)
        new_files = sorted(after_files - before_files)

        if not new_files:
            print(
                f"[WARN] No new metrics files detected in {runs_dir} after "
                f"running label='{br.logical_run.label}', backend='{br.backend}'.",
                file=sys.stderr,
            )
            continue

        print(f"[INFO] Detected {len(new_files)} new metrics file(s) in {runs_dir}:")
        for path in new_files:
            print(f"       - {path}")

        # For each new metrics file, extract run_id and write perf_<run_id>.txt
        for metrics_file in new_files:
            try:
                run_id = extract_run_id(metrics_file)
            except ValueError as e:
                print(f"[WARN] {e}", file=sys.stderr)
                continue

            perf_dest = perf_dir / f"perf_{run_id}.txt"
            try:
                perf_dest.write_text(combined_perf_text, encoding="utf-8")
            except OSError as e:
                print(
                    f"[ERROR] Failed to write perf output file {perf_dest}: {e}",
                    file=sys.stderr,
                )
                continue

            run_result = RunResult(
                logical_run=br.logical_run,
                backend=br.backend,
                run_id=run_id,
                metrics_file=metrics_file,
                perf_file=perf_dest,
            )
            results_list.append(run_result)

            label = br.logical_run.label
            if label not in results_map:
                results_map[label] = {}
            results_map[label][br.backend] = run_result

            print(
                f"[INFO] Recorded RunResult: label='{label}', backend='{br.backend}', "
                f"run_id='{run_id}'"
            )

    return results_list, results_map, perf_temp_files
