import cv2
import numpy as np
from PIL import Image
from waifuset import logging

LOGGER = logging.get_logger("outpaint")


def get_random_outpainting_mask(
    image: Image.Image,
    min_margin_ratio: float = 0.1,
    max_margin_ratio: float = 0.4,
):
    margin_ratio = np.random.uniform(min_margin_ratio, max_margin_ratio)
    width, height = image.size

    # Randomly select 1 to 4 sides to extend
    sides = np.random.choice(['top', 'bottom', 'left', 'right'], size=np.random.randint(1, 5), replace=False)
    mask = np.ones((height, width), dtype=np.uint8) * 255  # Create a white mask
    for side in sides:
        if side == 'top':
            margin = int(height * margin_ratio)
            mask[:margin, :] = 0
        elif side == 'bottom':
            margin = int(height * margin_ratio)
            mask[-margin:, :] = 0
        elif side == 'left':
            margin = int(width * margin_ratio)
            mask[:, :margin] = 0
        elif side == 'right':
            margin = int(width * margin_ratio)
            mask[:, -margin:] = 0

    return mask
