import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
from typing import Any, Dict, Union
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_project_root(root):
    global PROJECT_ROOT
    PROJECT_ROOT = Path(root)
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    return PROJECT_ROOT


def set_device(device):
    global DEVICE
    DEVICE = torch.device(device)
    return DEVICE


def open_as_pil(image: Union[Dict[str, Any], np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, dict):
        return Image.open(image['image_path'])
    elif isinstance(image, Image.Image):
        return image
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image)
    else:
        raise ValueError("Unsupported image type. Must be a dict, PIL Image, or numpy array.")


def open_as_numpy(image: Union[Dict[str, Any], np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, dict):
        return cv2.imread(image['image_path'])
    elif isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError("Unsupported image type. Must be a dict, PIL Image, or numpy array.")


def random_rotate(
    image: Union[Dict[str, Any], np.ndarray, Image.Image],
    min_angle: float = 0,
    max_angle: float = 360,
) -> Image.Image:
    """
    Randomly rotate an image within a specified angle range.

    Args:
        image: A dictionary containing image data or a PIL Image.
        angle_range: A tuple specifying the range of angles for rotation.

    Returns:
        A rotated PIL Image object.
    """
    angle = np.random.uniform(min_angle, max_angle)
    pil_image = open_as_pil(image)
    return pil_image.rotate(angle, expand=True)


def random_scale(
    image: Union[Dict[str, Any], np.ndarray, Image.Image],
    min_scale: float = 0.8,
    max_scale: float = 1.2,
    resample: int = Image.Resampling.LANCZOS,
) -> Image.Image:
    scale = np.random.uniform(min_scale, max_scale)
    pil_image = open_as_pil(image)
    width, height = pil_image.size
    new_size = (int(width * scale), int(height * scale))
    return pil_image.resize(new_size, resample=resample)


def random_transparency(
    image: Union[Dict[str, Any], np.ndarray, Image.Image],
    min_alpha: float = 0.5,
    max_alpha: float = 1.0,
) -> Image.Image:
    """
    Randomly adjust the transparency of an image.

    Args:
        image: A dictionary containing image data or a PIL Image.
        min_alpha: Minimum alpha value (transparency).
        max_alpha: Maximum alpha value (transparency).

    Returns:
        A PIL Image object with adjusted transparency.
    """
    alpha = np.random.uniform(min_alpha, max_alpha)
    pil_image = open_as_pil(image)
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')
    alpha_channel = pil_image.split()[-1]
    alpha_channel = alpha_channel.point(lambda p: p * alpha)
    return Image.merge('RGBA', (*pil_image.split()[:-1], alpha_channel))


def random_blur(
    image: Union[Dict[str, Any], np.ndarray, Image.Image],
    min_radius: float = 0.0,
    max_radius: float = 5.0,
) -> Image.Image:
    """
    Randomly apply Gaussian blur to an image.

    Args:
        image: A dictionary containing image data or a PIL Image.
        min_radius: Minimum blur radius.
        max_radius: Maximum blur radius.

    Returns:
        A blurred PIL Image object.
    """
    radius = np.random.uniform(min_radius, max_radius)
    pil_image = open_as_pil(image)
    return pil_image.filter(ImageFilter.GaussianBlur(radius))


def random_addition(
    image: Union[Dict[str, Any], np.ndarray, Image.Image],
    content_image: Union[Dict[str, Any], np.ndarray, Image.Image],
    allow_beyond_bounds: bool = False,
) -> Image.Image:
    image = open_as_pil(image)
    content_image = open_as_pil(content_image)
    if not allow_beyond_bounds:
        content_image = content_image.crop(
            (0, 0, min(content_image.width, image.width), min(content_image.height, image.height))
        )
    x_offset = np.random.randint(0, image.width - content_image.width + 1)
    y_offset = np.random.randint(0, image.height - content_image.height + 1)
    image.paste(content_image, (x_offset, y_offset), content_image.convert("RGBA"))
    return image


def remove_transparency_border(image: Image.Image) -> Image.Image:
    if image.mode != 'RGBA':
        return image
    img_array = np.array(image)
    alpha = img_array[:, :, 3]

    # Find rows and columns that are not completely transparent
    rows_with_content = np.where(alpha.any(axis=1))[0]
    cols_with_content = np.where(alpha.any(axis=0))[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        return image  # completely transparent image

    # Get bounding box coordinates
    top = rows_with_content[0]
    bottom = rows_with_content[-1]
    left = cols_with_content[0]
    right = cols_with_content[-1]

    # Crop the image
    cropped_image = image.crop((left, top, right + 1, bottom + 1))

    return cropped_image
