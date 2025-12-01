# Ray-Tracing-CPU
CE1114 – Proyecto de Aplicación de la Ingeniería en Computadores

Ray tracer escrito en Rust con dos backends de cómputo:

- **Scalar**: implementación escalar “clásica”.
- **Neon**: implementación optimizada usando SIMD (`core::arch::aarch64`) en Raspberry Pi 5.

Además del renderizado de imágenes, el proyecto genera **métricas internas** del ray tracer y **métricas de hardware** utilizando `perf`.

---

## 1. Requisitos

### 1.1. Sistema operativo

- Linux de 64 bits. Probado en:
    - Raspberry Pi 5 (Ubuntu / Raspberry Pi OS 64-bit).


### 1.2. Paquetes básicos

Asegurarse de tener herramientas de compilación:

```bash
sudo apt update
sudo apt install -y build-essential git
```


### 1.3. Rust y Cargo

Instalar rustup
```bash
sudo apt install rustup
```

Recargar el entorno (si hace falta):
```bash
source $HOME/.cargo/env
```

Verificar instalación
```bash
rustup default stable
rustc --version
cargo --version
```

El proyecto asume Rust estable (stable) en arquitectura aarch64-unknown-linux-gnu para el Raspberry Pi 5.


### 1.4. Herramientas para métricas de hardware (perf)

En Raspberry Pi es necesario instalar los paquetes de herramientas del kernel.

```bash
sudo apt update
sudo apt install -y linux-raspi-tools-6.8.0-1014
```

En otras distros podría ser:
```bash
sudo apt install -y linux-tools-$(uname -r)
```

En algunos casos se necesita permiso para usar perf sin ser root:
```bash
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```



## 2. Clonar y compilar el proyecto

### 2.1. Clonar el repositorio
```bash
git clone https://github.com/rubsalas/Ray-Tracing-CPU
cd Ray-Tracing-CPU
cd raytracer
```

### 2.2. Compilar en modo debug (desarrollo)
```bash
cargo build
```

### 2.3. Compilar en modo release (para mediciones)
```bash
cargo build --release
```


## 3. Ejecución con JSON de configuración

El binario puede recibir un archivo JSON que describe una o varias corridas.


### 3.1. Configuración de una sola corrida

Estos archivos se recomiendan que estén dentro de la carpeta config/.

Ejemplo de archivo config/neon_simple.json:

```json
{
    "runs": [
      {
        "backend": "neon",
        "scene": "simple",
        "image_width": 400,
        "aspect_ratio": 1.7777778,
        "samples_per_pixel": 100,
        "max_depth": 50,
        "scene_seed": 1000
      }
    ]
  }
```

Significado de los campos:

backend: backend de renderizado.
- "scalar" → implementación escalar.
- "neon" → implementación SIMD NEON.

scene: nombre lógico de la escena (por ejemplo "simple").
- "simple" → escena simple con tres esferas.
- "many_spheres" → escena compleja con tres grandes esferas y multiples pequeñas.

image_width: ancho de la imagen en píxeles.

aspect_ratio: relación de aspecto (ancho / alto). El alto se deriva de image_width / aspect_ratio.

samples_per_pixel: número de rayos de muestra por píxel (afecta ruido/calidad).

max_depth: profundidad máxima de rebotes por rayo.

scene_seed: semilla para el generador de números aleatorios (reproducibilidad).

```bash
cargo run --release config/neon_simple.json
```

Esto genera, para esa corrida:

- Una imagen en formato PPM en el folder output/.

- Un archivo de métricas de texto en el directorio runs/.


### 3.2. Configuración de varias corridas (runs_example.json)

También se pueden definir varias corridas en un solo archivo, por ejemplo config/multiple_runs.json:

```json
{
  "runs": [
    {
      "backend": "scalar",
      "scene": "many_spheres",
      "image_width": 400,
      "aspect_ratio": 1.7777778,
      "samples_per_pixel": 100,
      "max_depth": 50,
      "scene_seed": 1000
    },
    {
      "backend": "neon",
      "scene": "many_spheres",
      "image_width": 400,
      "aspect_ratio": 1.7777778,
      "samples_per_pixel": 100,
      "max_depth": 50,
      "scene_seed": 1000
    }
  ]
}
```

Al ejecutar:

```bash
cargo run --release config/multiple_runs.json
```

el binario recorre las corridas definidas en runs[] y para cada run:

a. Renderiza la imagen con esos parámetros y la exporta en output/.

b. Genera un archivo de métricas de texto en runs/.


### 3.3. Ejecución sin pasar un JSON

Si se ejecuta simplemente:

```bash
cargo run --release
```

El programa usa internamente una configuración predefinida equivalente a cargar un JSON por defecto (con backend, resolución, escena y parámetros fijados en el archivo config/runs.json).
Es decir, aunque no se pase un archivo .json en la línea de comandos, la ejecución sigue el mismo flujo lógico que una corrida definida, solo que con una configuración predeterminada.



## 4. Archivos de salida del ray tracer (Rust)

### 4.1 Imagenes

En cada corrida, el ray tracer genera la imagen en un archivo .ppm en el directorio output/, que está al mismo nivel del cual se ejecuta el binario.

El nombre del archivo está basado en la fecha y hora de corrida, el backend escogido, el tamaño de la imagen y la cantidad de samples per pixel.

Usualmente linux puede abrir este tipo de imagenes en el explorador de archivos, pero si no se pueden manejar con herramientas como:

```bash
feh 20251130_134110_neon_400x224_spp100.ppm
# o convertirlos a PNG:
convert 20251130_165833_neon_400x224_spp100.ppm 20251130_165833_neon_400x224_spp100.png
```


### 4.2 Métricas internas del ray tracer (carpeta runs/)

Cada corrida genera un archivo de metricas .txt en la carpeta runs/.

Un ejemplo de contenido real seria:
```text
run_id: 20251130_134110_neon_400x224_spp100
backend: Neon
image_width: 400
image_height: 224
samples_per_pixel: 100
max_depth: 50
scene_seed: 1000

[timing]
render_duration_ms: 4919

[neon_stats]
primary_rays_total: 8960000
primary_rays_accelerated: 5045580
primary_rays_fallback: 3914420

[scene_spheres]
scene_spheres_total: 5
scene_spheres_lambertian: 1
scene_spheres_metal: 2
scene_spheres_dielectric: 2

[core_stats]
core_rays_primary: 8960000
core_scalar_intersection_tests: 36353045
core_scalar_intersection_hits: 6850806
core_scalar_intersection_misses: 29502239
core_simd_intersection_tests: 44800000
core_simd_intersection_hits_lanes: 5045580
core_simd_intersection_misses_lanes: 3914420
core_lambertian_calls: 4902143
core_metal_calls: 593810
core_dielectric_calls: 765697
core_dielectric_reflect: 115853
core_dielectric_refract: 649844
```

Explicación de cada bloque

Cabecera:

- run_id: identificador único de la corrida.Normalmente incluye fecha/hora, backend, resolución y samples por píxel. Sirve para relacionar: imagen, métricas internas y métricas de perf.

- backend: backend usado (Neon o Scalar).

- image_width / image_height: resolución final en píxeles.

- samples_per_pixel: número de muestras por píxel.

- max_depth: profundidad máxima de rayos (rebotes).

- scene_seed: semilla utilizada en la generación de la escena / aleatoriedad.


[timing]

- render_duration_ms: tiempo total de render en milisegundos (solo CPU del render principal).


[neon_stats] (solo si se usa backend Neon)

- primary_rays_total: número total de rayos primarios (cámara). Suele ser: image_width * image_height * samples_per_pixel.

- primary_rays_accelerated: rayos primarios procesados por el camino acelerado NEON.

- primary_rays_fallback: rayos primarios que terminaron usando el camino escalar (por límites de alineamiento, borde de escena, etc.).


[scene_spheres]

- scene_spheres_total: cantidad total de esferas en la escena.

- scene_spheres_lambertian: esferas con material difuso.

- scene_spheres_metal: esferas metálicas.

- scene_spheres_dielectric: esferas dieléctricas (vidrio, etc.).

Sirve para documentar la complejidad geométrica y material de la escena.


[core_stats]

- core_rays_primary: igual que primary_rays_total; número de rayos de cámara lanzados.

- core_scalar_intersection_tests: cantidad de pruebas de intersección en código escalar.

- core_scalar_intersection_hits: número de intersecciones exitosas en el camino escalar.

- core_scalar_intersection_misses: número de pruebas escalar que no dieron intersección.

- core_simd_intersection_tests: cantidad de pruebas vectoriales (NEON) lanzadas.

- core_simd_intersection_hits_lanes: hits a nivel de lanes SIMD (cuántos lanes tuvieron intersección).

- core_simd_intersection_misses_lanes: lanes que no encontraron intersección.

- core_lambertian_calls: llamadas al shader/BSDF lambertiano.

- core_metal_calls: llamadas al material metálico.

- core_dielectric_calls: llamadas al material dieléctrico.

- core_dielectric_reflect: veces que la ruta dieléctrica resultó en reflexión.

- core_dielectric_refract: veces que la ruta dieléctrica resultó en refracción.


Con estos contadores se puede analizar:

a. Cuánto trabajo hace el camino escalar vs NEON.

b. Cómo se distribuyen los materiales.

c. Cuántos hits/misses por tipo de intersección.



## 5. Métricas de hardware con perf (run_with_perf.py)

Además de las métricas internas del ray tracer, el proyecto incluye un script para envolver el binario con perf y capturar contadores de hardware.


### 5.1. Ubicación del script

En la misma carpeta base del proyecto raytracer/ incluye una carpeta scripts/ donde está un archivo run_with_perf.py que hace esta función.


### 5.2. Eventos de perf que se miden

El script llama a:

```bash
perf stat -e <eventos> ...
```

con un conjunto de eventos como:

Básicos:

- task-clock

- context-switches

- cpu-migrations

- page-faults

- cycles

- instructions

- branches

- branch-misses

Cachés:

- l1d_cache, l1d_cache_refill

- l1i_cache, l1i_cache_refill

- l2d_cache, l2d_cache_refill

- l3d_cache, l3d_cache_refill

- ll_cache_rd, ll_cache_miss_rd


### 5.3. Uso básico del script

Desde raytracer/ (usa el binario release por defecto):

```bash
python3 scripts/run_with_perf.py config/neon_simple.json
```

Uso con varias configuraciones de corridas:

```bash
python3 scripts/run_with_perf.py config/multiple_runs.json
```

Parámetros principales del script:

- configs: uno o varios archivos JSON de configuración (tipo config/....json).

- -b, --binary: ruta al binario del ray tracer (por defecto ./target/release/raytracer).

- -r, --runs-dir: carpeta donde el ray tracer guarda las métricas internas (runs/ por defecto).

- --perf-dir: carpeta donde se guardan las salidas de perf (por ejemplo perf_metrics/). [Recomendado agregarlo siempre]

Ejemplo típico:

```bash
python3 scripts/run_with_perf.py \
    --binary ./target/release/raytracer \
    --runs-dir runs \
    --perf-dir perf_metrics \
    config/multiple_runs.json
```



## 6. Archivos de salida de perf y su interpretación


### 6.1. Ejemplo de salida de perf + métricas derivadas

```text
# started on Sun Nov 30 13:41:10 2025


 Performance counter stats for '/home/user/GitHub/Ray-Tracing-CPU/raytracer/target/release/raytracer /home/user/GitHub/Ray-Tracing-CPU/raytracer/config/neon_simple.json':

           4932.11 msec task-clock                       #    0.999 CPUs utilized             
               412      context-switches                 #   83.534 /sec                      
                 0      cpu-migrations                   #    0.000 /sec                      
               609      page-faults                      #  123.476 /sec                      
       11787674937      cycles                           #    2.390 GHz                         (49.95%)
       27013891616      instructions                     #    2.29  insn per cycle              (49.97%)
        2260330961      branches                         #  458.288 M/sec                       (49.99%)
          18906434      branch-misses                    #    0.84% of all branches             (50.01%)
        9274865554      l1d_cache                        #    1.881 G/sec                       (50.03%)
            514551      l1d_cache_refill                 #  104.327 K/sec                       (50.05%)
        7335198376      l1i_cache                        #    1.487 G/sec                       (50.06%)
           2316534      l1i_cache_refill                 #  469.684 K/sec                       (42.90%)
           3822408      l2d_cache                        #  775.004 K/sec                       (42.88%)
            253229      l2d_cache_refill                 #   51.343 K/sec                       (42.86%)
            408418      l3d_cache                        #   82.808 K/sec                       (42.84%)
            158093      l3d_cache_refill                 #   32.054 K/sec                       (42.82%)
            330414      ll_cache_rd                      #   66.992 K/sec                       (42.81%)
            159979      ll_cache_miss_rd                 #   32.436 K/sec                       (42.81%)

       4.934586359 seconds time elapsed

       4.918494000 seconds user
       0.013992000 seconds sys

# ==== Derived metrics (computed by run_with_perf.py) ====
branch_miss_rate: 0.008364 (0.836%)
l1d_miss_rate: 0.000055 (0.006%)
l1d_mpki: 0.0190 misses / kilo-instructions
l1i_miss_rate: 0.000316 (0.032%)
l2d_miss_rate: 0.066249 (6.625%)
l2d_mpki: 0.0094 misses / kilo-instructions
llc_miss_rate: 0.484177 (48.418%)
llc_mpki: 0.0059 misses / kilo-instructions
ipc: 2.2917 instructions / cycle
inst_per_sec: 5474398389.39 instructions / second
```

### 6.2. Explicación de los contadores principales

Tiempo y uso de CPU

- task-clock (msec): tiempo de CPU consumido por el proceso.

- seconds time elapsed: tiempo de reloj de pared (wall-clock).

- seconds user: tiempo en modo usuario.

- seconds sys: tiempo en modo kernel.

- CPUs utilized: proporción de CPUs utilizadas (cerca de 1.0 significa un núcleo completo ocupado todo el tiempo).

Branching

- branches: número total de instrucciones de salto condicional/indirecto.

- branch-misses: número de saltos mal predichos.

- % of all branches: tasa de mispredictions.

Caché L1 data e instruction

- l1d_cache: accesos a la caché L1 de datos.

- l1d_cache_refill: refills de L1D (cuando L1D falla y necesita traer desde niveles inferiores).

- l1i_cache: accesos a la caché L1 de instrucciones.

- l1i_cache_refill: refills de L1I.

Caché L2 data

- l2d_cache: accesos a la caché L2 de datos.

- l2d_cache_refill: refills de L2D.

L3 / Last-Level Cache (LLC)

- l3d_cache: accesos a L3/unified cache.

- l3d_cache_refill: refills de L3.

- ll_cache_rd: lecturas que llegan al último nivel de caché.

- ll_cache_miss_rd: lecturas que fallan también en el último nivel (y van a memoria).


### 6.3. Métricas derivadas calculadas por el script .py

El script toma los contadores principales y calcula, entre otras:

IPC (Instrucciones por ciclo)
```text
ipc = instructions / cycles
```

Throughput de instrucciones
```text
inst_per_sec = instructions / time_elapsed
```

Branch miss rate
```text
branch_miss_rate = branch-misses / branches
```

Tasas de misses de caché (miss rate)

- L1 data:
```text
l1d_miss_rate = l1d_cache_refill / l1d_cache
```

- L1 instruction:
```text
l1i_miss_rate = l1i_cache_refill / l1i_cache
```

- L2 data:
```text
l2d_miss_rate = l2d_cache_refill / l2d_cache
```

- Last Level Cache:
```text
llc_miss_rate = ll_cache_miss_rd / ll_cache_rd
```

MPKI (Misses por mil instrucciones)

- L1 data:
```text
l1d_mpki = 1000 * l1d_cache_refill / instructions
```

- L2 data:
```text
l2d_mpki = 1000 * l2d_cache_refill / instructions
```

- LLC:
```text
llc_mpki = 1000 * ll_cache_miss_rd / instructions
```

MPKI ayuda a comparar configuraciones con diferente carga de trabajo, porque normaliza los misses por cantidad de trabajo realizado (instrucciones), no por segundo.
