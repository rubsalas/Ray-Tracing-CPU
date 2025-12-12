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
            "run": [
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
        config/

    then the generated configs will be:
        config/exp_simple_meta_simple_400x226_spp10_seed3526_scalar.json
        config/exp_simple_meta_simple_400x226_spp10_seed3526_neon.json

    Returns:
    --------
    backend_runs:
        A list of BackendRun instances describing each concrete backend
        configuration (including the path to the generated JSON file).

    temp_files:
        A list of Paths to all generated JSON files. This is useful for
        implementing the --delete-temp behavior later, so that the main
        function can remove them at the end if requested.
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
        # meta_stem comes from the original meta-config file name.
        meta_stem = lr.meta_config_path.stem

        for backend in BACKENDS:
            # Create a deep copy of the logical run's config so that we can
            # inject the backend field without mutating the original.
            config_entry = deepcopy(lr.config_data)
            config_entry["backend"] = backend

            config_wrapper = {"run": [config_entry]}

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
