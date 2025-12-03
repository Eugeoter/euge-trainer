import cv2
import numpy as np
import time
from PIL import Image
from typing import Any, Dict, Union
from . import utils
from .dataclass import ImageEditionData


def get_addition_data(
    orig_img: Union[Dict[str, Any], np.ndarray, Image.Image],
    content_img: Union[Dict[str, Any], np.ndarray, Image.Image],
    size_alignment: bool = True,
    remove_transparency_border: bool = True,
    scale: float = 1.0,
    transparency: float = 1.0,
    rotate: float = 0.0,
):
    orig_img = utils.open_as_pil(orig_img)
    content_img = utils.open_as_pil(content_img)

    if remove_transparency_border:
        content_img = utils.remove_transparency_border(content_img)
    if size_alignment:
        orig_size = orig_img.size
        content_size = content_img.size
        if orig_size != content_size:
            content_img = content_img.resize(orig_size, resample=Image.Resampling.LANCZOS)
    if scale != 1.0:
        content_img = utils.random_scale(content_img, min_scale=scale, max_scale=scale)
    if transparency != 1.0:
        content_img = utils.random_transparency(content_img, min_alpha=transparency, max_alpha=transparency)
    if rotate != 0.0:
        content_img = utils.random_rotate(content_img, rotate, rotate)

    x


def get_random_addition_data(
    orig_img: Union[Dict[str, Any], np.ndarray, Image.Image],
    content_img: Union[Dict[str, Any], np.ndarray, Image.Image],
    size_alignment: bool = True,
    remove_transparency_border: bool = True,
    random_scale: bool = True,
    prob_random_scale: float = 1.0,
    random_transparency: bool = True,
    prob_random_transparency: float = 0.25,
    random_rotate: bool = True,
    prob_random_rotate: float = 0.25,
):
    orig_img = utils.open_as_pil(orig_img)
    content_img = utils.open_as_pil(content_img)

    if remove_transparency_border:
        content_img = utils.remove_transparency_border(content_img)
    if size_alignment:
        orig_size = orig_img.size
        content_size = content_img.size
        if orig_size != content_size:
            content_img = content_img.resize(orig_size, resample=Image.Resampling.LANCZOS)
    if random_scale and np.random.rand() < prob_random_scale:
        content_img = utils.random_scale(content_img, min_scale=0.05, max_scale=0.5)
    if random_transparency and np.random.rand() < prob_random_transparency:
        content_img = utils.random_transparency(content_img, min_alpha=0.25, max_alpha=1.0)
    if random_rotate and np.random.rand() < prob_random_rotate:
        content_img = utils.random_rotate(content_img, 0, 360)

    tar_img = utils.random_addition(
        orig_img,
        content_img,
        allow_beyond_bounds=True,
    )

    return ImageEditionData(
        target_image=tar_img,
        reference_images=orig_img,
    )


def get_random_removal_data(
    orig_img: Union[Dict[str, Any], np.ndarray, Image.Image],
    content_img: Union[Dict[str, Any], np.ndarray, Image.Image],
    size_alignment: bool = True,
    remove_transparency_border: bool = True,
    random_scale: bool = True,
    prob_random_scale: float = 1.0,
    random_transparency: bool = True,
    prob_random_transparency: float = 0.25,
    random_rotate: bool = True,
    prob_random_rotate: float = 0.25,
):
    addition_data = get_random_addition_data(
        orig_img=orig_img,
        content_img=content_img,
        size_alignment=size_alignment,
        remove_transparency_border=remove_transparency_border,
        random_scale=random_scale,
        prob_random_scale=prob_random_scale,
        random_transparency=random_transparency,
        prob_random_transparency=prob_random_transparency,
        random_rotate=random_rotate,
        prob_random_rotate=prob_random_rotate,
    )
    return ImageEditionData(
        target_image=addition_data.reference_images[0],
        reference_images=addition_data.target_image
    )


def get_random_replacement_data(
    orig_img: Union[Dict[str, Any], np.ndarray, Image.Image],
    content_img: Union[Dict[str, Any], np.ndarray, Image.Image],
    size_alignment: bool = True,
    remove_transparency_border: bool = True,
    random_scale: bool = True,
    prob_random_scale: float = 1.0,
    random_transparency: bool = True,
    prob_random_transparency: float = 0.25,
    random_rotate: bool = True,
    prob_random_rotate: float = 0.25,
):
