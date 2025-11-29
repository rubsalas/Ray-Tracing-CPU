mod ray;
mod vec3;
mod camera;
mod interval;
mod prelude;

mod image;
mod world;

mod render;
mod metrics;
mod scene;
mod run;

mod simd;


use std::io::Result as IoResult;

use crate::scene::SceneKind;
use crate::render::BackendKind;
use crate::run::{RunConfig, execute_run};

fn main() -> IoResult<()> {
    let base_seed = 1000_u64;
    
    // Vector de RunConfigs
    let configs = vec![

        // Aquí se define una sola corrida
        RunConfig {
            backend: BackendKind::Scalar,
            scene: SceneKind::ManySpheres,      // Simple or ManySpheres
            image_width: 403,               // 1200
            aspect_ratio: 16.0 / 9.0,
            samples_per_pixel: 100,         // 500
            max_depth: 50,
            scene_seed: Some(base_seed), // o None si no se quiere fijar
        },
        
        // Esto es otra corrida
        RunConfig {
            backend: BackendKind::Neon,
            scene: SceneKind::ManySpheres,      // Simple or ManySpheres,
            image_width: 403,               // 1200
            aspect_ratio: 16.0 / 9.0,
            samples_per_pixel: 100,         // 500
            max_depth: 50,
            scene_seed: Some(base_seed), // mismo seed para comparar backends
        }      

    ];

    // Hace que se ejecuten todas las corridas
    for cfg in configs {
        execute_run(&cfg)?;
    }

    Ok(())
}
