use actix_web::error::ErrorInternalServerError;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use chrono::SecondsFormat::Micros;
use chrono::{DateTime, Utc};
use google_cloud_storage::client::{Client, ClientConfig};
use google_cloud_storage::http::objects::download::Range;
use google_cloud_storage::http::objects::get::GetObjectRequest;
use image::ImageReader;
use ndarray::Array;
use ort::{GraphOptimizationLevel, Session, Value};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::cmp::Ordering;
use std::sync::Arc;

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
    bucket_name: String,
    image_path: String,
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
    client: Client,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let session = Arc::new(
        Session::builder()
            .expect("Failed to acquire session builder.")
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .expect("Failed to set optimization level.")
            .with_intra_threads(2) // intra op thread pool must have at least one thread for RunAsync
            .expect("Failed to set intra_threads.")
            .with_inter_threads(1)
            .expect("Failed set inter_threads.")
            .commit_from_file("model.onnx")
            .expect("Failed to set commit from file."),
    );

    let client = Client::new(
        ClientConfig::default()
            .with_auth()
            .await
            .expect("Failed to auth client."),
    );
    let app_state = web::Data::new(AppState { session, client });

    println!("Starting server on http://0.0.0.0:8082");
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/", web::get().to(health_endpoint))
            .route("/predict/", web::post().to(prediction_endpoint))
    })
    .bind(("0.0.0.0", 8082))?
    .workers(4)
    .run()
    .await?;

    Ok(())
}

async fn health_endpoint() -> impl Responder {
    let healthy = Healthy::default();
    HttpResponse::Ok().json(healthy)
}

async fn prediction_endpoint(
    app_state: web::Data<AppState>,
    payload: web::Json<Payload>,
) -> Result<HttpResponse, actix_web::Error> {
    let downloading_image_st = Utc::now();
    let contents = {
        let get_request = GetObjectRequest {
            bucket: payload.bucket_name.clone(),
            object: payload.image_path.clone(),
            ..Default::default()
        };
        app_state
            .client
            .download_object(&get_request, &Range::default())
            .await
            .map_err(ErrorInternalServerError)?
    };
    log_event("async-downloading-image", downloading_image_st)?;
    let preprocessing_and_model_inference_st = Utc::now();
    let prediction = {
        let preprocessing_image_st = Utc::now();
        let img = ImageReader::new(std::io::Cursor::new(contents))
            .with_guessed_format()
            .map_err(ErrorInternalServerError)?
            .decode()
            .map_err(ErrorInternalServerError)?
            .to_rgb8();

        let (width, height) = img.dimensions();
        let img_data_f32: Vec<f32> = img.into_raw().iter().map(|&x| x as f32).collect();
        let img_array =
            Array::from_shape_vec((1, 3, height as usize, width as usize), img_data_f32)
                .map_err(ErrorInternalServerError)?;

        let input_tensor = ort::Tensor::from_array(img_array).map_err(ErrorInternalServerError)?;
        let inputs: Vec<(String, Value)> = vec![("input".to_string(), Value::from(input_tensor))];
        log_event("preprocessing-image", preprocessing_image_st)?;
        let model_inference_st = Utc::now();
        let output_tensors = app_state
            .session
            .run_async(inputs)
            .map_err(ErrorInternalServerError)?
            .await
            .map_err(ErrorInternalServerError)?;

        let output_array: ndarray::ArrayViewD<'_, f32> = output_tensors[0]
            .try_extract_tensor::<f32>()
            .map_err(ErrorInternalServerError)?;

        let max_prob_idx = output_array
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| ErrorInternalServerError("Prediction failed"))?;
        log_event("model-inference", model_inference_st)?;
        match max_prob_idx {
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
        }
    };
    log_event(
        "preprocessing-and-model-inference",
        preprocessing_and_model_inference_st,
    )?;
    let response = Response { prediction };
    Ok(HttpResponse::Ok().json(response))
}

fn log_event(name: &str, start_time: DateTime<Utc>) -> Result<(), actix_web::Error> {
    let event = json!({
        "name": name,
        "start_time": start_time.to_rfc3339_opts(Micros, true),
        "end_time":  Utc::now().to_rfc3339_opts(Micros, true),
    });
    println!(
        "{}",
        serde_json::to_string(&event).map_err(ErrorInternalServerError)?
    );
    Ok(())
}
