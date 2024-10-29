extern crate core;

use actix_web::error::ErrorInternalServerError;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use ndarray::{Array, Array4};
use ort::{GraphOptimizationLevel, Session, Value};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::sync::Arc;
use std::thread::available_parallelism;

#[derive(Serialize, Deserialize, Debug)]
struct Healthy {
    status: String,
    message: String,
}

impl Default for Healthy {
    fn default() -> Self {
        Healthy {
            status: "healthy".to_string(),
            message: "Rust ONNX model service is running.".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Payload {
    image: String,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
enum WeatherClass {
    Dew,
    Fogsmog,
    Frost,
    Glaze,
    Hail,
    Lightning,
    Rain,
    Rainbow,
    Rime,
    Sandstorm,
    Snow,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Response {
    prediction: WeatherClass,
}

struct AppState {
    session: Arc<Session>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let session = Arc::new(
        Session::builder()
            .expect("Failed to acquire session builder.")
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .expect("Failed to set optimization level.")
            .with_intra_threads(3) // intra op thread pool must have at least one thread for RunAsync
            .expect("Failed to set intra_threads.")
            .with_inter_threads(1)
            .expect("Failed set inter_threads.")
            .commit_from_file("model.onnx")
            .expect("Failed to set commit from file."),
    );
    let app_state = web::Data::new(AppState { session });
    println!(
        "Application will use default number of workers: {}",
        available_parallelism()
            .expect("Failed to retrieve available parallelism.")
            .get()
    );

    println!("Starting server on http://0.0.0.0:8082");
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/", web::get().to(health_endpoint))
            .route("/predict/", web::post().to(prediction_endpoint))
    })
    .bind(("0.0.0.0", 8082))?
    .run()
    .await?;

    Ok(())
}

async fn health_endpoint() -> impl Responder {
    let healthy = Healthy::default();
    HttpResponse::Ok().json(healthy)
}

#[inline]
fn process_image(raw_data: &[u8], width: u32, height: u32) -> Array4<f32> {
    // Pre-allocate the output array with the correct shape
    let mut output = Array::zeros((1, 3, height as usize, width as usize));
    // Process RGB channels in one pass
    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel_start = (y * width as usize + x) * 3;
            output[[0, 0, y, x]] = raw_data[pixel_start] as f32; // R
            output[[0, 1, y, x]] = raw_data[pixel_start + 1] as f32; // G
            output[[0, 2, y, x]] = raw_data[pixel_start + 2] as f32; // B
        }
    }
    output
}

async fn prediction_endpoint(
    app_state: web::Data<AppState>,
    payload: web::Json<Payload>,
) -> Result<HttpResponse, actix_web::Error> {
    let image_bytes = STANDARD
        .decode(payload.image.as_bytes()) // Use as_bytes() to avoid clone
        .map_err(ErrorInternalServerError)?;

    let img = image::load_from_memory_with_format(&image_bytes, image::ImageFormat::Jpeg)
        .map_err(ErrorInternalServerError)?
        .to_rgb8();

    let (width, height) = img.dimensions();
    let img_array = process_image(img.as_raw(), width, height);

    let input_tensor = ort::Tensor::from_array(img_array).map_err(ErrorInternalServerError)?;
    let inputs: Vec<(String, Value)> = vec![("input".to_string(), Value::from(input_tensor))];

    let output_tensors = app_state
        .session
        .run_async(inputs)
        .map_err(ErrorInternalServerError)?
        .await
        .map_err(ErrorInternalServerError)?;

    let output_array = output_tensors[0]
        .try_extract_tensor::<f32>()
        .map_err(ErrorInternalServerError)?;

    let prediction = match output_array
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx)
        .ok_or_else(|| ErrorInternalServerError("Prediction failed"))?
    {
        0 => WeatherClass::Dew,
        1 => WeatherClass::Fogsmog,
        2 => WeatherClass::Frost,
        3 => WeatherClass::Glaze,
        4 => WeatherClass::Hail,
        5 => WeatherClass::Lightning,
        6 => WeatherClass::Rain,
        7 => WeatherClass::Rainbow,
        8 => WeatherClass::Rime,
        9 => WeatherClass::Sandstorm,
        10 => WeatherClass::Snow,
        _ => return Err(ErrorInternalServerError("Invalid class index")),
    };

    Ok(HttpResponse::Ok().json(Response { prediction }))
}
