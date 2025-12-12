import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import LogicalRun


def _sanitize_label(raw: str) -> str:
    """
    Sanitize a label string so that it is safe to use as part of a filename.

    The goal here is to:
        - Remove or replace characters that are problematic in file paths
          (such as '/', '\\', whitespace).
        - Preserve as much semantic information as possible.
    """
    label = "_".join(raw.split())  # Replace whitespace with underscores.

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
    """
    scene = entry.get("scene")
    if scene is None:
        base = f"run{index}"
    else:
        base = str(scene)

    width = entry.get("image_width")
    aspect = entry.get("aspect_ratio")

    res_component: Optional[str] = None
    if isinstance(width, (int, float)):
        if isinstance(aspect, (int, float)) and aspect != 0:
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
                "backend": "scalar"                          # will be ignored
            },
            ...
        ]
    }

    Notes:
    ------
    - The 'backend' field (if present) is ignored and a warning is printed,
      because perf_compare always runs both scalar and neon backends for
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

    warned_backend_once = False
    used_labels: Dict[str, int] = {}

    for idx, entry in enumerate(runs):
        if not isinstance(entry, dict):
            print(
                f"[ERROR] Invalid entry in 'run' array at index {idx} in {path}: "
                f"each run must be a JSON object.",
                file=sys.stderr,
            )
            sys.exit(1)

        entry_copy: Dict[str, Any] = dict(entry)

        # Handle 'backend' if present in the meta-config (ignore but warn).
        if "backend" in entry_copy:
            if not warned_backend_once:
                print(
                    f"[WARN] Field 'backend' found in meta-config {path}. "
                    f"It will be ignored; perf_compare always runs both "
                    f"scalar and neon backends.",
                    file=sys.stderr,
                )
                warned_backend_once = True
            entry_copy.pop("backend", None)

        # Determine label: use 'label' if present, otherwise generate one.
        raw_label = entry_copy.get("label")
        if isinstance(raw_label, str) and raw_label.strip():
            label = _sanitize_label(raw_label.strip())
            entry_copy.pop("label", None)
        else:
            label = _generate_label_from_entry(entry_copy, idx)

        # Ensure uniqueness of labels within the same file.
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
