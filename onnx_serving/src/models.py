from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

WeatherClass = Literal[
    "dew",
    "fogsmog",
    "frost",
    "glaze",
    "hail",
    "lightning",
    "rain",
    "rainbow",
    "rime",
    "sandstorm",
    "snow",
]
WeatherClasses = list[WeatherClass]
CLASSES: WeatherClasses = [
    "dew",
    "fogsmog",
    "frost",
    "glaze",
    "hail",
    "lightning",
    "rain",
    "rainbow",
    "rime",
    "sandstorm",
    "snow",
]


class AliasedBaseModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Healthy(BaseModel):
    status: str = Field(default="healthy")
    message: str = Field(default="Python ONNX model service is running.")


class Payload(AliasedBaseModel):
    image: str


class Response(AliasedBaseModel):
    prediction: WeatherClass
