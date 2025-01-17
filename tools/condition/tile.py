import numpy as np
from PIL import Image
from typing import Literal
from waifuset import logging

CONTROL_TYPE = "tile"
LOGGER = logging.get_logger(CONTROL_TYPE)

INTERPOLATIONS = [
    Image.Resampling.LANCZOS,
    Image.Resampling.BICUBIC,
    Image.Resampling.BILINEAR,
    Image.Resampling.NEAREST,
    Image.Resampling.HAMMING,
    Image.Resampling.BOX,
]

RESIZING_RATIOS = [16, 8, 4, 2, 1]


def get_tile(
    img: Image.Image,
    k: int = None,
    interpolation_downscale: Literal[Image.Resampling.LANCZOS, Image.Resampling.BICUBIC, Image.Resampling.BILINEAR, Image.Resampling.NEAREST, Image.Resampling.HAMMING, Image.Resampling.BOX] = None,
    interpolation_upscale: Literal[Image.Resampling.LANCZOS, Image.Resampling.BICUBIC, Image.Resampling.BILINEAR, Image.Resampling.NEAREST, Image.Resampling.HAMMING, Image.Resampling.BOX] = None,
    division: int = 64,
) -> Image.Image:
    if k is None:
        k = np.random.choice(RESIZING_RATIOS)
    if interpolation_downscale is None:
        interpolation_downscale = np.random.choice(INTERPOLATIONS)
    if interpolation_upscale is None:
        interpolation_upscale = np.random.choice(INTERPOLATIONS)
    orig_width, orig_height = img.size
    target_width = int(orig_width // k // division * division)
    target_height = int(orig_height // k // division * division)
    target_width = max(division, target_width)
    target_height = max(division, target_height)
    img = img.resize((target_width, target_height), resample=interpolation_downscale)
    img = img.resize((orig_width, orig_height), resample=interpolation_upscale)
    return img
