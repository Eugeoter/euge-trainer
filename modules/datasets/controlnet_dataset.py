import torch
import io
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Any, Callable, Literal
from waifuset import logging
from controlnet_aux.processor import Processor
from .t2i_dataset import T2IDataset
from ..utils import dataset_utils

CONTROL_IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ]
)

CNAUX_PROCESSORS = {}


class ControlNetDataset(T2IDataset):
    control_image_type: Literal[
        "canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
        "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
        "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
        "scribble_hed", "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
        "softedge_pidinet", "softedge_pidsafe", "dwpose"
    ] = None
    control_image_getter: Callable = lambda self, img_md, *args, **kwargs: None
    control_image_getter_kwargs: Dict[str, Any] = {}
    control_image_resampling: str = 'lanczos'
    cache_control_image: bool = False
    control_image_cache_dir: str = None
    keep_control_image_in_memory: bool = False

    def check_config(self):
        if not self.control_image_getter_kwargs:
            self.control_image_getter_kwargs = {}

        if self.control_image_type is not None:
            if self.control_image_getter is not None:
                self.logger.info(f"Overwrite control image getter with control image type: {logging.yellow(self.control_image_type)}")
            else:
                self.logger.info(f"Using control image type: {logging.yellow(self.control_image_type)}")

            if self.control_image_getter_kwargs and 'control_type' in self.control_image_getter_kwargs:
                self.logger.warning(f"Overwrite control image getter kwargs control_type with control image type: {logging.yellow(self.control_image_type)}")
            self.logger.info(f"Control image getter kwargs: {self.control_image_getter_kwargs}")

        if self.cache_control_image:
            if not self.control_image_cache_dir:
                raise ValueError("Control image cache dir is not set")

        if self.control_image_cache_dir:
            if not self.cache_control_image:
                self.logger.warning("Control image cache dir is set but cache control image is not enabled, control image will not be cached")
            elif not self.control_image_getter and not self.control_image_type:
                self.logger.warning("Control image cache dir is set but control image getter or type is not set, control image will not be cached")
            else:
                self.logger.info(f"Control image cache dir: {logging.yellow(self.control_image_cache_dir)}")
                os.makedirs(self.control_image_cache_dir, exist_ok=True)

    def get_control_image_cache_path(self, img_md):
        if not self.control_image_cache_dir:
            return None
        return os.path.join(self.control_image_cache_dir, f"{img_md['image_key']}.png")

    def open_control_image(self, img_md) -> Image.Image:
        if self.cache_control_image and (control_image_cache_path := self.get_control_image_cache_path(img_md)) is not None and os.path.exists(control_image_cache_path) and (control_image := Image.open(control_image_cache_path)) is not None and control_image.verify():
            pass
        elif self.control_image_type and (control_image := get_controlnet_aux_condition(self.get_image(img_md), control_type=self.control_image_type)) is not None:
            if self.control_image_cache_dir:
                if not os.path.exists(control_image_cache_path):
                    control_image.save(control_image_cache_path)
        elif self.control_image_getter and (control_image := (self.control_image_getter(img_md, **self.control_image_getter_kwargs))) is not None:
            if isinstance(control_image, Image.Image):
                pass
            elif isinstance(control_image, np.ndarray):
                control_image = Image.fromarray(control_image)
            else:
                raise ValueError(f"Control image must be a PIL Image or a numpy array, got {type(control_image)}, {control_image}")
            if self.control_image_cache_dir:
                if not os.path.exists(control_image_cache_path):
                    control_image.save(control_image_cache_path)
        elif (control_image := img_md.get('control_image')) is not None:
            if isinstance(control_image, Image.Image):
                pass
            elif isinstance(control_image, dict):
                if (control_image_bytes := control_image.get('bytes')) is not None:
                    control_image = Image.open(io.BytesIO(control_image_bytes))
                elif (control_image_path := control_image.get('path')) is not None:
                    control_image = Image.open(control_image_path)
                else:
                    raise ValueError(f"Control image not found for {img_md.get('image_key')}, control_image: {control_image}")

        elif (control_image_path := img_md.get('control_image_path')) is not None:
            control_image = Image.open(control_image_path)
        else:
            self.logger.warning(f"Control image not found for {img_md.get('image_key')}")
            return None
        assert isinstance(control_image, Image.Image), f"control image must be a PIL Image, got {type(control_image)}, {control_image}"
        return control_image

    def get_control_image(self, img_md, type: Literal['pil', 'tensor', 'numpy'] = 'tensor') -> torch.Tensor:
        control_image = self.open_control_image(img_md)
        if control_image is None:
            return None
        control_image = control_image.convert('RGB')
        image_size, _, bucket_size = self.get_size(img_md, update=True)

        crop_ltrb = self.get_crop_ltrb(img_md, update=True)
        control_image = dataset_utils.resize_if_needed(control_image, image_size, resampling=self.control_image_resampling)
        control_image = dataset_utils.crop_ltrb_if_needed(control_image, crop_ltrb)
        control_image = dataset_utils.resize_if_needed(control_image, bucket_size, resampling=self.control_image_resampling)
        if type == 'tensor':
            control_image = CONTROL_IMAGE_TRANSFORMS(control_image)
        elif type == 'numpy':
            control_image = np.array(control_image)
        elif type == 'pil':
            pass
        else:
            raise ValueError(f"Invalid control image type: {type}, must be 'pil', 'tensor' or 'numpy'")
        return control_image

    def get_control_image_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict[str, Any]:
        sample = dict(
            control_images=[],
        )
        for i, img_key in enumerate(batch):
            img_md = self.dataset[img_key]
            control_image = self.get_control_image(img_md)
            is_flipped = samples['is_flipped'][i]
            if is_flipped:
                control_image = torch.flip(control_image, dims=[2])
            sample["control_images"].append(control_image)
        sample["control_images"] = torch.stack(sample["control_images"], dim=0).to(memory_format=torch.contiguous_format).float()
        return sample


def get_controlnet_aux_condition(
    image: Image.Image,
    control_type: Literal[
        "canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
        "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
        "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
        "scribble_hed", "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
        "softedge_pidinet", "softedge_pidsafe", "dwpose"
    ],
    **kwargs
) -> Image.Image:
    r"""
    Get the condition of an image using controlnet_aux library.
    """
    global CNAUX_PROCESSORS
    # options are:
    # ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
    #  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
    #  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
    #  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
    #  "softedge_pidinet", "softedge_pidsafe", "dwpose"]
    if control_type not in CNAUX_PROCESSORS:
        CNAUX_PROCESSORS[control_type] = Processor(control_type, params=kwargs)
    processor = CNAUX_PROCESSORS[control_type]
    condition: Image.Image = processor(image, to_pil=True)
    if isinstance(image, Image.Image):
        target_width, target_height = image.size
    elif isinstance(image, torch.Tensor):
        target_height, target_width = image.shape[-2:]
    elif isinstance(image, np.ndarray):
        target_height, target_width = image.shape[:2]
    else:
        raise ValueError(f"Invalid image type. Expected PIL Image, torch.Tensor or numpy array, got {type(image)}")
    if isinstance(condition, np.ndarray):
        condition = Image.fromarray(condition)
    if condition.width != target_width or condition.height != target_height:
        condition = condition.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
    return condition
