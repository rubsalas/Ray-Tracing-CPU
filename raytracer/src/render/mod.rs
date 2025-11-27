// src/render/mod.rs

pub mod scalar;
pub mod neon;
pub mod primary;

use crate::camera::Camera;
use crate::prelude::*;               // Color, etc.
use crate::world::hittable::Hittable;
use crate::world::sphere::Sphere;

use scalar::ScalarRenderer;
use neon::NeonRenderer;

use std::any::Any;

/// Tipo de backend de render que se va a utilizar.
#[derive(Clone, Copy, Debug)]
pub enum BackendKind {
    /// Camino completamente escalar.
    Scalar,
    /// Camino con extensiones NEON (SIMD) en las partes que se vayan portando.
    Neon,
}

/// Parámetros lógicos del render, compartidos entre backends.
pub struct RenderParams {
    /// Ancho de la imagen en píxeles.
    pub image_width: i32,
    /// Relación de aspecto (width / height).
    pub aspect_ratio: f64,
    /// Muestras por píxel.
    pub samples_per_pixel: i32,
    /// Profundidad máxima de rebotes del rayo.
    pub max_depth: i32,
}

/// Trait que define la interfaz común de un backend de render.
pub trait Renderer {
    /// Renderiza la escena `world` usando la cámara `camera` y los parámetros `params`,
    /// dejando el resultado en `framebuffer` (un Color por píxel, orden fila por fila).
    fn render(
        &mut self,
        world: &dyn Hittable,
        camera: &mut Camera,
        params: &RenderParams,
        framebuffer: &mut [Color],
    );

    /// Permite hacer downcast desde &mut dyn Renderer a tipos concretos
    /// (por ejemplo, NeonRenderer) para leer estadísticas internas.
    fn as_any(&self) -> &dyn Any;
}

/// Construye un renderer para una escena dada.
///
/// - `backend`: qué backend se desea (Scalar / Neon).
/// - `sphere_accel`: para NEON, un `Vec<Sphere>` plano que se usará en las
///   rutinas vectorizadas de intersección (world_hit4_spheres).
///
/// Nota:
/// - En el backend escalar, `sphere_accel` se ignora.
/// - En el backend NEON, se pasa tal cual al constructor de `NeonRenderer`.
pub fn make_renderer_for_scene(
    backend: BackendKind,
    sphere_accel: Option<Vec<Sphere>>,
) -> Box<dyn Renderer> {
    match backend {
        BackendKind::Scalar => Box::new(ScalarRenderer::new()),
        BackendKind::Neon   => Box::new(NeonRenderer::new(sphere_accel)),
    }
}

/// Wrapper de compatibilidad.
///
/// Todo el código que antes hacía:
///
/// ```rust
/// let mut renderer = make_renderer(backend, None);
/// ```
///
/// sigue funcionando, porque esta función simplemente delega en
/// `make_renderer_for_scene`.
pub fn make_renderer(
    backend: BackendKind,
    sphere_accel: Option<Vec<Sphere>>,
) -> Box<dyn Renderer> {
    make_renderer_for_scene(backend, sphere_accel)
}
