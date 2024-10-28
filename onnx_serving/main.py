from io import BytesIO

import numpy as np
import onnxruntime as rt
from aiohttp import ClientSession, ClientTimeout
from fastapi import FastAPI
from gcloud.aio.storage import Bucket, Storage
from PIL import Image
from src.models import CLASSES, Healthy, Payload, Response

# FastAPI application initialization
app = FastAPI()

# High timeouts for load testing
CLIENT_TIMEOUT = ClientTimeout(total=3600, connect=3600, sock_read=3600, sock_connect=3600)
TIMEOUT = 3600

# Creating ONNX session for inference
sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
# Set ONNXRuntime to use one thread per worker to prevent CPU oversubscription
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
session = rt.InferenceSession(f"model.onnx", sess_options)


@app.get("/")
async def health_endpoint() -> Healthy:
    return Healthy()


@app.post(path="/predict/")
async def prediction_endpoint(payload: Payload) -> Response:
    async with ClientSession(timeout=CLIENT_TIMEOUT) as client_session:
        async with Storage(session=client_session) as client:
            bucket = Bucket(client, payload.bucket_name)
            blob = await bucket.get_blob(blob_name=payload.image_path, timeout=TIMEOUT, session=client_session)
            contents = await blob.download(timeout=TIMEOUT, session=client_session)
    img = Image.open(BytesIO(contents)).convert("RGB")
    img_tensor = np.array(img, dtype=np.float32).transpose(2, 0, 1)  # Transpose from HWC to CHW format
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_tensor})
    max_prob_idx = np.argmax(outputs[0], axis=1)[0]
    return Response(prediction=CLASSES[max_prob_idx])
