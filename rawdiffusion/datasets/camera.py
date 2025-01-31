from dataclasses import dataclass
from typing import Optional


@dataclass
class Camera:
    dataset: str
    name: str
    min_value: int
    black_level: int
    white_level: int


cameras = [
    Camera("FiveK", "NIKON_D700", min_value=0, black_level=0, white_level=16383),
    Camera(
        "FiveK", "Canon_EOS_5D", min_value=0, black_level=0, white_level=4095
    ),  # black level is 127 before preprocessing
    Camera("NOD", "Nikon750", min_value=284, black_level=600, white_level=16383),
    Camera("NOD", "SonyRX100m7", min_value=0, black_level=800, white_level=16380),
]


def normalize_name(name):
    return name.lower().replace("_", "")


def compare_names(name1, name2):
    return normalize_name(name1) == normalize_name(name2)


def get_camera(dataset, name) -> Optional[Camera]:
    dataset = dataset.lower()
    name = name.lower()

    for camera in cameras:
        if compare_names(camera.dataset, dataset) and compare_names(camera.name, name):
            return camera

    return None
