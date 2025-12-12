#!/usr/bin/env python3
"""
Script para lanzar corridas del ray tracer usando archivos JSON de configuración
y envolver cada ejecución con `perf stat`.

Además, parsea la salida de perf para calcular métricas derivadas como:
- ipc
- branch_miss_rate
- l1d_miss_rate, l1i_miss_rate, l2d_miss_rate, llc_miss_rate
- l1d_mpki, l2d_mpki, llc_mpki
- inst_per_sec

Uso típico:

    # Ejecutar el binario release con un archivo de configuración:
    # (el binario espera el path al JSON como primer argumento)
    python3 scripts/run_with_perf.py config/runs_example.json

    # Ejecutar con binario explícito y varias configuraciones:
    python3 scripts/run_with_perf.py \
        --binary ./target/release/raytracer \
        config/exp_simple.json config/exp_many.json
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

# Eventos que se le piden a perf.
# Incluye los contadores básicos + los de caché que estamos usando.
PERF_EVENTS = [
    # básicos
    "task-clock",
    "context-switches",
    "cpu-migrations",
    "page-faults",
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    # cachés
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


def parse_args() -> argparse.Namespace:
    """Se definen y se parsean los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Ejecutar el ray tracer envuelto en `perf stat`, usando uno o varios "
            "archivos JSON de configuración."
        )
    )

    parser.add_argument(
        "configs",
        nargs="+",
        help="Rutas a archivos JSON de configuración (por ejemplo config/runs_example.json).",
    )

    parser.add_argument(
        "-b",
        "--binary",
        default="./target/release/raytracer",
        help=(
            "Ruta al binario compilado del ray tracer. "
            "Por defecto: ./target/release/raytracer"
        ),
    )

    parser.add_argument(
        "-r",
        "--runs-dir",
        default="runs",
        help=(
            "Directorio donde el ray tracer escribe las métricas de salida "
            "(archivos con run_id). Por defecto: runs"
        ),
    )

    parser.add_argument(
        "--perf-dir",
        default=None,
        help=(
            "Directorio donde se guardan los archivos perf_*.txt generados por este script. "
            "Por defecto usa el mismo valor de --runs-dir."
        ),
    )

    parser.add_argument(
        "--perf",
        default="perf",
        help="Nombre o ruta del ejecutable `perf`. Por defecto: perf",
    )

    return parser.parse_args()


def list_metric_files(runs_dir: Path) -> Set[Path]:
    """
    Se obtienen todos los archivos de métricas existentes en el directorio `runs_dir`.

    Se asume que los archivos de métricas son .txt.
    """
    if not runs_dir.exists():
        return set()
    return set(runs_dir.glob("*.txt"))


def extract_run_id(metrics_file: Path) -> str:
    """
    Se extrae el valor de `run_id` desde un archivo de métricas.

    Se busca la primera línea que empiece con 'run_id:' y se devuelve
    el valor a la derecha. Si no se encuentra, se lanza una excepción.
    """
    with metrics_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("run_id:"):
                # Ejemplo de línea: "run_id: 20251130_015158_neon_403x226_spp100"
                return line.split(":", 1)[1].strip()

    raise ValueError(f"No se encontró 'run_id:' en el archivo {metrics_file}")


def parse_perf_metrics(perf_text: str) -> Dict[str, float]:
    """
    Parsea la salida de `perf stat` y devuelve un diccionario metric_name -> valor.

    Se soportan:
    - Líneas tipo:
          11764165449      cycles
          9273841591      l1d_cache
    - task-clock en msec:
          4920.93 msec task-clock
    - tiempos al final:
          4.923556772 seconds time elapsed
          4.911326000 seconds user
          0.009996000 seconds sys
    """
    metrics: Dict[str, float] = {}

    for line in perf_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue

        parts = stripped.split()

        # Líneas de tiempo al final
        # Ej: "4.923556772 seconds time elapsed"
        if len(parts) >= 4 and parts[1] == "seconds" and parts[2] == "time" and parts[3] == "elapsed":
            try:
                metrics["time_elapsed"] = float(parts[0])
            except ValueError:
                pass
            continue

        # Ej: "4.911326000 seconds user" / "0.009996000 seconds sys"
        if len(parts) >= 3 and parts[1] == "seconds" and parts[2] in ("user", "sys"):
            key = f"time_{parts[2]}"
            try:
                metrics[key] = float(parts[0])
            except ValueError:
                pass
            continue

        # Resto de métricas: esperamos algo tipo:
        #   11764165449      cycles  ...
        #   4920.93 msec task-clock ...
        first = parts[0]
        # Debe empezar con dígito o punto
        if not (first[0].isdigit() or first[0] == "."):
            continue

        value_str = first.replace(",", "")
        try:
            value = float(value_str)
        except ValueError:
            continue

        metric_name: Optional[str] = None

        # task-clock viene con unidad "msec"
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


def compute_derived_metrics(m: Dict[str, float]) -> Dict[str, float]:
    """
    A partir de los contadores crudos en `m`, calcula métricas derivadas.
    Solo se incluye una métrica si tiene los datos necesarios.
    """
    def get(name: str) -> Optional[float]:
        return m.get(name)

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

    # branch miss rate
    bmr = safe_div(branch_misses, branches)
    if bmr is not None:
        derived["branch_miss_rate"] = bmr

    # Miss rates de caché
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

    # Throughput de instrucciones
    ips = safe_div(instructions, time_elapsed)
    if ips is not None:
        derived["inst_per_sec"] = ips

    return derived


def format_derived_metrics(derived: Dict[str, float]) -> str:
    """
    Devuelve un bloque de texto legible con las métricas derivadas,
    ordenadas por “capas”:

    1) Branch
    2) L1 (data / instruction)
    3) L2
    4) LLC
    5) Throughput / IPC
    6) Cualquier métrica adicional que aparezca
    """
    lines: List[str] = []
    lines.append("# ==== Derived metrics (computed by run_with_perf.py) ====")

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

    # 5) Throughput / IPC
    if "ipc" in derived:
        val = derived["ipc"]
        lines.append(f"ipc: {val:.4f} instructions / cycle")
        printed.add("ipc")

    if "inst_per_sec" in derived:
        val = derived["inst_per_sec"]
        lines.append(f"inst_per_sec: {val:.2f} instructions / second")
        printed.add("inst_per_sec")

    # 6) Cualquier métrica adicional no contemplada arriba
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


def run_with_perf(
    perf_cmd: str,
    binary_path: Path,
    config_path: Path,
    runs_dir: Path,
    perf_dir: Path,
) -> None:
    """
    Se ejecuta una corrida del ray tracer envuelta en `perf stat`.

    - `runs_dir`: donde el ray tracer deja sus métricas (con run_id).
    - `perf_dir`: donde este script deja perf_tmp_*.txt y perf_<run_id>.txt.

    Pasos:
    1. Se guarda la lista de archivos de métricas existentes antes de la corrida.
    2. Se ejecuta `perf stat -e <EVENTOS> -o perf_tmp_*.txt -- <binary> <config>`.
    3. Se obtienen los archivos de métricas nuevos creados en `runs_dir`.
    4. Se lee el `run_id` de cada nuevo archivo.
    5. Se copia la salida de perf a `perf_dir/perf_<run_id>.txt` por cada run_id detectado,
       añadiendo al final un bloque con métricas derivadas.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    # Se toma la foto de los archivos de métricas antes de la corrida.
    before_files = list_metric_files(runs_dir)

    # Se construye un nombre temporal para el archivo de salida de perf.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    perf_tmp = perf_dir / f"perf_tmp_{timestamp}.txt"

    # Construimos la lista de eventos para perf (-e).
    events_arg = ",".join(PERF_EVENTS)

    # Se construye el comando completo para perf.
    cmd: List[str] = [
        perf_cmd,
        "stat",
        "-e",
        events_arg,
        "-o",
        str(perf_tmp),
        "--",
        str(binary_path),
        str(config_path),
    ]

    print(f"[INFO] Ejecutando perf para config: {config_path}")
    print(f"[INFO] Comando: {' '.join(cmd)}")

    try:
        # Se ejecuta el comando y se espera a que termine.
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] La ejecución de perf o del binario falló para {config_path}: {e}",
            file=sys.stderr,
        )
        # Se decide no borrar perf_tmp para poder inspeccionarlo si algo falló.
        return

    # Leemos la salida de perf para calcular métricas derivadas.
    try:
        perf_text = perf_tmp.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[ERROR] No se pudo leer {perf_tmp}: {e}", file=sys.stderr)
        return

    metrics = parse_perf_metrics(perf_text)
    derived = compute_derived_metrics(metrics)
    derived_block = format_derived_metrics(derived) if derived else ""

    combined_perf_text = perf_text
    if derived_block:
        combined_perf_text = perf_text.rstrip() + "\n\n" + derived_block + "\n"

    # Se obtienen los archivos de métricas después de la corrida.
    after_files = list_metric_files(runs_dir)

    # Se identifican los archivos nuevos creados en esta ejecución.
    new_files = sorted(after_files - before_files)

    if not new_files:
        print(
            f"[WARN] No se detectaron nuevos archivos de métricas en {runs_dir} "
            f"después de ejecutar {config_path}."
        )
        return

    print(f"[INFO] Se detectan {len(new_files)} archivo(s) de métricas nuevo(s):")
    for path in new_files:
        print(f"       - {path}")

    # Se extrae el run_id de cada archivo nuevo y se copia la salida de perf.
    for metrics_file in new_files:
        try:
            run_id = extract_run_id(metrics_file)
        except ValueError as e:
            print(f"[WARN] {e}", file=sys.stderr)
            continue

        perf_dest = perf_dir / f"perf_{run_id}.txt"

        # Se escribe el contenido original de perf + métricas derivadas.
        perf_dest.write_text(combined_perf_text, encoding="utf-8")

        print(f"[INFO] Se escribe salida de perf (con métricas derivadas) en: {perf_dest}")

    # Se puede borrar el archivo temporal si ya no se necesita.
    try:
        perf_tmp.unlink()
    except OSError:
        # Si falla el borrado no es crítico; se deja el archivo temporal.
        print(f"[WARN] No se pudo borrar archivo temporal {perf_tmp}", file=sys.stderr)


def main() -> None:
    """Función principal del script."""
    args = parse_args()

    binary_path = Path(args.binary).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    perf_dir = Path(args.perf_dir).resolve() if args.perf_dir else runs_dir

    if not binary_path.exists():
        print(f"[ERROR] No se encuentra el binario en: {binary_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Binario: {binary_path}")
    print(f"[INFO] Directorio de métricas del raytracer (runs-dir): {runs_dir}")
    print(f"[INFO] Directorio de métricas perf (perf-dir): {perf_dir}")
    print(f"[INFO] Ejecutable perf: {args.perf}")
    print(f"[INFO] Eventos perf: {', '.join(PERF_EVENTS)}")
    print()

    # Se recorre cada archivo de configuración proporcionado.
    for config_str in args.configs:
        config_path = Path(config_str).resolve()

        if not config_path.exists():
            print(
                f"[ERROR] No se encuentra el archivo de configuración: {config_path}",
                file=sys.stderr,
            )
            continue

        run_with_perf(args.perf, binary_path, config_path, runs_dir, perf_dir)
        print()

    print("[INFO] Finaliza ejecución de todas las configuraciones.")


if __name__ == "__main__":
    main()
