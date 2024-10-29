import json
from base64 import b64encode
from io import BytesIO
from pathlib import Path

from PIL import Image

# The desired image must be in images/ directory
IMAGE_PATH = "images/rime_5868.jpg"

if __name__ == "__main__":
    image_path = Path(IMAGE_PATH)
    json_path = image_path.with_suffix(".json")
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    encoded_string = b64encode(buffered.getvalue()).decode("utf-8")
    payload = {"image": encoded_string}
    json_path.write_text(json.dumps(payload))
    print(f"Payload saved to: {json_path}")
