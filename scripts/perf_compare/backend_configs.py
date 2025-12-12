import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

from .models import BACKENDS, BackendRun, LogicalRun


def generate_backend_configs(
    logical_runs: List[LogicalRun],
    output_dir: Path,
) -> Tuple[List[BackendRun], List[Path]]:
    """
    For each LogicalRun, generate one backend-specific JSON config per backend
    in BACKENDS (currently: scalar and neon).

    The generated JSON has the structure:
        {
            "runs": [
                { ...original_entry_fields..., "backend": "<backend>" }
            ]
        }

    and is written to a file located under `output_dir`, using the following
    naming scheme:

        <meta_stem>_<label>_<backend>.json

    Where:
        - meta_stem is the stem (filename without extension) of the
          originating meta-config (which typically lives in scripts/config/).
        - label is the LogicalRun label (sanitized for filenames).
        - backend is one of the values in BACKENDS.

    Example:
    --------
    If the meta-config is:
        scripts/config/exp_simple_meta.json
    and the logical run has:
        label = "simple_400x226_spp10_seed3526"

    and output_dir is:
        raytracer/config/

    then the generated configs will be:
        raytracer/config/exp_simple_meta_simple_400x226_spp10_seed3526_scalar.json
        raytracer/config/exp_simple_meta_simple_400x226_spp10_seed3526_neon.json
    """
    backend_runs: List[BackendRun] = []
    temp_files: List[Path] = []

    # Ensure the output directory exists.
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(
            f"[ERROR] Failed to create or access output config directory {output_dir}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    for lr in logical_runs:
        meta_stem = lr.meta_config_path.stem

        for backend in BACKENDS:
            # Deep copy of the logical run config so we don't mutate the original.
            config_entry = deepcopy(lr.config_data)
            config_entry["backend"] = backend

            # 🔹 IMPORTANT: the ray tracer expects "runs", not "run"
            config_wrapper = {"runs": [config_entry]}

            filename = f"{meta_stem}_{lr.label}_{backend}.json"
            config_path = output_dir / filename

            try:
                with config_path.open("w", encoding="utf-8") as f:
                    json.dump(config_wrapper, f, indent=2)
            except OSError as e:
                print(
                    f"[ERROR] Failed to write backend config {config_path}: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)

            temp_files.append(config_path)

            backend_runs.append(
                BackendRun(
                    logical_run=lr,
                    backend=backend,
                    config_path=config_path,
                )
            )

    return backend_runs, temp_files
