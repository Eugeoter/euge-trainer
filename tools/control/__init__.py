import cv2
import numpy as np
from PIL import Image
from typing import Literal
from .utils import *

CNAUX_PROCESSORS = {}
CNAUX_CONTROL_TYPES = [
    "canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
    "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
    "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
    "scribble_hed", "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
    "softedge_pidinet", "softedge_pidsafe", "dwpose"
]
ADDITIONAL_CONTROL_TYPES = [
    "random_canny", "animalpose", "densepose", "teed", "mteed", "tile", "mobile_sam", "random_blur"
]


def get_controlnet_aux_condition(
    img_md,
    control_type: Literal[
        "canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
        "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
        "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
        "scribble_hed", "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
        "softedge_pidinet", "softedge_pidsafe", "dwpose",
        "random_canny", "animalpose", "densepose", "teed", "mteed", "tile", "mobile_sam", "random_blur"
    ],
    **kwargs
) -> np.ndarray:
    r"""
    Get the condition of an image using controlnet_aux library.
    """
    from controlnet_aux.processor import Processor
    # Options are:
    # ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
    #  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
    #  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
    #  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
    #  "softedge_pidinet", "softedge_pidsafe", "dwpose"]
    if control_type in ADDITIONAL_CONTROL_TYPES:
        if control_type == "random_canny":
            return get_random_canny_condition(img_md)
        elif control_type == "animalpose":
            return get_animalpose_condition(img_md)
        elif control_type == "densepose":
            return get_densepose_condition(img_md)
        elif control_type == "teed":
            return get_teed_condition(img_md)
        elif control_type == "mteed":
            return get_mteed_condition(img_md)
        elif control_type == "tile":
            return get_tile_condition(img_md, **kwargs)
        elif control_type == "mobile_sam":
            return get_mobile_sam_condition(img_md, **kwargs)
        elif control_type == 'random_blur':
            return get_random_blur_condition(img_md, **kwargs)
    elif control_type in CNAUX_CONTROL_TYPES:
        if control_type not in CNAUX_PROCESSORS:
            CNAUX_PROCESSORS[control_type] = Processor(control_type, params=kwargs)
        processor = CNAUX_PROCESSORS[control_type]
        img_path = img_md['image_path']
        image = Image.open(img_path)
        condition = processor(image, to_pil=True)
        return np.array(condition)
    else:
        raise ValueError(f"Invalid control type: {control_type}. Expected one of {', '.join(CNAUX_CONTROL_TYPES + ADDITIONAL_CONTROL_TYPES)}")


def get_mlsd_condition(img_md, thr_v=0.5, thr_d=0.5) -> np.ndarray:
    from .mlsd import get_mlsd
    img_path = img_md['image_path']
    img = cv2.imread(img_path)
    mlsd_condition = get_mlsd(img, thr_v, thr_d)
    return mlsd_condition


def get_random_canny_condition(img_md, low_threshold=None, high_threshold=None) -> np.ndarray:
    from .canny import get_canny
    if (img_path := img_md.get('image_path')) is not None:
        image = cv2.imread(img_path)
    elif (image := img_md.get('image')) is not None:
        pass
    else:
        raise ValueError("No image found in metadata.")

    if not isinstance(image, np.ndarray):
        image = np.array(image)
    canny_condition = get_canny(image, low_threshold, high_threshold)
    return canny_condition


def get_hed_condition(img_md) -> np.ndarray:
    from .hed import get_hed
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    hed_condition = get_hed(image)
    return hed_condition


def get_midas_depth_condition(img_md, a=np.pi * 2.0, bg_th=0.1) -> np.ndarray:
    from .midas import get_midas_depth
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    midas_condition = get_midas_depth(image, a, bg_th)
    if midas_condition.shape[:2] != image.shape[:2]:
        midas_condition = cv2.resize(midas_condition, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return midas_condition


def get_midas_normal_condition(img_md, a=np.pi * 2.0, bg_th=0.1) -> np.ndarray:
    from .midas import get_midas_normal
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    midas_condition = get_midas_normal(image, a, bg_th)
    if midas_condition.shape[:2] != image.shape[:2]:
        midas_condition = cv2.resize(midas_condition, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return midas_condition


def get_openpose_condition(img_md, include_body=True, include_hand=False, include_face=False) -> np.ndarray:
    from .openpose import get_openpose
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    openpose_condition = get_openpose(image, include_body=include_body, include_hand=include_hand, include_face=include_face)
    return openpose_condition


def get_dwpose_condition(img_md, include_body=True, include_hand=False, include_face=False) -> np.ndarray:
    from .openpose import get_dwpose
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    openpose_condition = get_dwpose(image, include_body=include_body, include_hand=include_hand, include_face=include_face)
    return openpose_condition


def get_animalpose_condition(img_md, include_body=True, include_hand=False, include_face=False) -> np.ndarray:
    from .openpose import get_animalpose
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    openpose_condition = get_animalpose(image, include_body=include_body, include_hand=include_hand, include_face=include_face)
    return openpose_condition


def get_densepose_condition(img_md, cmap='viridis') -> np.ndarray:
    from .densepose import get_densepose
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    densepose_condition = get_densepose(image, cmap=cmap)
    return densepose_condition


def get_sketch_condition(img_md) -> np.ndarray:
    from .sketch import get_sketch
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    sketch_condition = get_sketch(image)
    return sketch_condition


def get_lineart_condition(img_md, model_name='sk_model.pth', white_bg=False) -> np.ndarray:
    from .lineart import get_lineart
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    lineart_condition = get_lineart(image, model_name=model_name, white_bg=white_bg)
    return lineart_condition


def get_manga_line_condition(img_md) -> np.ndarray:
    from .manga_line import get_manga_line
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    manga_line_condition = get_manga_line(image)
    return manga_line_condition


def get_pidinet_condition(img_md, is_safe=False, apply_fliter=False) -> np.ndarray:
    from .pidinet import get_pidinet
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    scribble_pidinet_condition = get_pidinet(image, is_safe=is_safe, apply_fliter=apply_fliter)
    return scribble_pidinet_condition


def get_teed_condition(img_md) -> np.ndarray:
    from .teed import get_teed
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    teeds_condition = get_teed(image, mteed=False)
    return teeds_condition


def get_mteed_condition(img_md) -> np.ndarray:
    from .teed import get_teed
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    teeds_condition = get_teed(image, mteed=True)
    return teeds_condition


def get_mobile_sam_condition(img_md, detect_resolution=512, image_resolution=512) -> np.ndarray:
    from .mobile_sam import get_mobile_sam
    img_path = img_md['image_path']
    image = cv2.imread(img_path)
    sam_condition = get_mobile_sam(image, detect_resolution=detect_resolution, image_resolution=image_resolution)
    return sam_condition


def get_tile_condition(img_md, k=None, interpolation_downscale=None, interpolation_upscale=None) -> np.ndarray:
    from .tile import get_tile
    img_path = img_md['image_path']
    image = Image.open(img_path)
    tile_condition = get_tile(image, k, interpolation_downscale, interpolation_upscale)
    return np.array(tile_condition)


def get_random_blur_condition(img_md, radius=None) -> np.ndarray:
    from .blur import get_blur
    img_path = img_md['image_path']
    image = Image.open(img_path)
    blur_condition = get_blur(image, radius)
    return np.array(blur_condition)
