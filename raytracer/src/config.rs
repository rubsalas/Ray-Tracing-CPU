// src/config.rs

use std::fs;
use std::io::{Error as IoError, ErrorKind, Result as IoResult};
use std::path::Path;

use serde::Deserialize;

use crate::run::RunConfig;

/// Struct raíz que representa el archivo JSON completo.
/// El JSON debe tener un objeto con una clave `runs` que contiene un arreglo.
#[derive(Debug, Deserialize)]
pub struct RunsConfigFile {
    pub runs: Vec<RunConfig>,
}

/// Función para cargar un archivo de configuración y devolver la lista de RunConfig.
///
/// `path`: ruta al archivo JSON con la configuración.
///
/// Se lee el archivo, se parsea el JSON y se devuelve el vector de corridas.
/// Si hay un error de lectura o de parseo, se devuelve un IoError descriptivo.
pub fn load_runs_from_file<P: AsRef<Path>>(path: P) -> IoResult<Vec<RunConfig>> {
    let path_ref = path.as_ref();

    // Se lee el archivo completo a memoria como String.
    let contents = fs::read_to_string(path_ref)?;

    // Se intenta parsear el JSON a RunsConfigFile.
    let parsed: RunsConfigFile = serde_json::from_str(&contents).map_err(|err| {
        // Se convierte el error de parseo en un IoError para mantener IoResult.
        IoError::new(
            ErrorKind::InvalidData,
            format!(
                "Error al parsear archivo de configuración {:?}: {}",
                path_ref, err
            ),
        )
    })?;

    // Se devuelve el vector de RunConfig.
    Ok(parsed.runs)
}
