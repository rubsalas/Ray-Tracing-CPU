mod ray;
mod run;
mod simd;
mod vec3;
mod image;
mod world;
mod scene;
mod camera;
mod config;
mod render;
mod metrics;
mod prelude;
mod interval;

use std::io::Result as IoResult;
use std::env;
use std::path::Path;

use crate::run::{RunConfig, execute_run};

fn main() -> IoResult<()> {
    // Se obtienen los argumentos de línea de comandos.
    // Ejemplo de uso:
    //     cargo run --release -- config/runs_example.json
    //
    // args[0] = nombre del binario
    // args[1] = ruta al archivo de configuración (si existe)
    let args: Vec<String> = env::args().collect();

    // Se determina la ruta del archivo de configuración.
    // Si el usuario pasa un argumento, se usa ese.
    // Si no pasa nada, se usa un archivo por defecto en `config/runs.json`.
    let config_path: &str = if args.len() >= 2 {
        &args[1]
    } else {
        "config/runs.json"
    };

    // Se imprime la ruta usada para facilitar depuración.
    eprintln!("Usando archivo de configuración: {}", config_path);

    // Se carga el archivo de configuración y se obtienen los RunConfig.
    let configs: Vec<RunConfig> = config::load_runs_from_file(Path::new(config_path))?;

    // Se ejecutan todas las corridas definidas en el archivo.
    for cfg in configs {
        execute_run(&cfg)?;
    }

    Ok(())
}
