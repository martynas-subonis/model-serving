from io import BytesIO

import torch
from aiohttp import ClientSession, ClientTimeout
from fastapi import FastAPI
from gcloud.aio.storage import Bucket, Storage
from PIL import Image
from src.models import CLASSES, Healthy, Payload, Response
from torch import device, load
from torch import max as torch_max
from torch.nn import Linear
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import Compose, Lambda

# FastAPI application initialization
app = FastAPI()

# High timeouts for load testing
CLIENT_TIMEOUT = ClientTimeout(total=3600, connect=3600, sock_read=3600, sock_connect=3600)
TIMEOUT = 3600

# Set PyTorch to use one thread per worker to prevent CPU oversubscription
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Initializing PyTorch model and transformer for inference
t_device = device("cpu")
model = mobilenet_v3_small()
model.classifier[-1] = Linear(model.classifier[-1].in_features, len(CLASSES))
model.load_state_dict(load("model.pth", weights_only=True))
model.to(t_device)
model.eval()
transform = Compose(
    [
        MobileNet_V3_Small_Weights.DEFAULT.transforms().to(t_device),
        Lambda(lambda x: x.unsqueeze(0)),  # Add batch dimension
    ]
)


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
    img_tensor = transform(img)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, max_prob_idx = torch_max(outputs, 1)
    return Response(prediction=CLASSES[max_prob_idx])
