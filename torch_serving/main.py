from base64 import b64decode
from io import BytesIO

import torch
from fastapi import FastAPI
from PIL import Image
from src.models import CLASSES, Healthy, Payload, Response
from torch import device, load
from torch import max as torch_max
from torch.nn import Linear
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import Compose, Lambda

# FastAPI application initialization
app = FastAPI()

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
    img = Image.open(BytesIO(b64decode(payload.image))).convert("RGB")
    img_tensor = transform(img)
    with torch.inference_mode():
        outputs = model(img_tensor)
        _, max_prob_idx = torch_max(outputs, 1)
    return Response(prediction=CLASSES[max_prob_idx])
