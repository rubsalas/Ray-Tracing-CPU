mod ray;
mod run;
mod vec3;
mod color;
mod scene;
mod camera;
mod output;
mod render;
mod sphere;
mod metrics;
mod prelude;
mod hittable;
mod interval;
mod material;
mod hittable_list;

use std::io::Result as IoResult;

use crate::scene::SceneKind;
use crate::render::BackendKind;
use crate::run::{RunConfig, execute_run};

fn main() -> IoResult<()> {
    
    // Vector de RunConfigs
    let configs = vec![

        // Aquí se define una sola corrida
        RunConfig {
            backend: BackendKind::Scalar,
            scene: SceneKind::ManySpheres,
            image_width: 400,       // 1200
            aspect_ratio: 16.0 / 9.0,
            samples_per_pixel: 100, // 500
            max_depth: 50,
        },

        // Esto es otra corrida
        RunConfig {
            backend: BackendKind::Neon,
            scene: SceneKind::Initial,
            image_width: 400,       // 1200
            aspect_ratio: 16.0 / 9.0,
            samples_per_pixel: 100, // 500
            max_depth: 50,
        }

    ];

    // Hace que se ejecuten todas las corridas
    for cfg in configs {
        execute_run(&cfg)?;
    }

    Ok(())
}
