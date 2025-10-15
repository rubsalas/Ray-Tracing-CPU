// main.rs
// Progreso por stderr y la imagen PPM por stdout.
// Se ve el progreso en la terminal y la imagen en el archivo
// cargo run > image.ppm

use std::io::{self, Write}; // para flush() de stderr

fn main() {

    let image_width: u32 = 256;
    let image_height: u32 = 256;

    // Encabezado PPM por stdout
    println!("P3");
    println!("{image_width} {image_height}");
    println!("255");

    for j in 0..image_height {

        // ---- Indicador de progreso ----
        // \r regresa el cursor al inicio de la línea y reescribe el texto.
        // eprint! escribe en stderr
        // flush() fuerza que se muestre de inmediato.

        let remaining = image_height - j;
        eprint!("\rScanlines remaining: {remaining} ");
        io::stderr().flush().unwrap();

        for i in 0..image_width {

            let r = (i as f64) / ((image_width - 1) as f64);
            let g = (j as f64) / ((image_height - 1) as f64);
            let b = 0.0_f64;

            let ir = (255.999 * r) as u32;
            let ig = (255.999 * g) as u32;
            let ib = (255.999 * b) as u32;

            println!("{ir} {ig} {ib}");
        }
    }
    
    // Mensaje final por stderr (con espacios para borrar restos de la línea anterior)
    eprintln!("\rDone.                         ");

}
