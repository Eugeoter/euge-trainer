import numpy as np
from PIL import Image, ImageFilter
from waifuset import logging

CONTROL_TYPE = "blur"
LOGGER = logging.get_logger(CONTROL_TYPE)


def get_blur(img, radius=None) -> Image.Image:
    if radius is None:
        radius = np.random.randint(1, 10)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img
