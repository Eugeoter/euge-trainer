import torch
import io
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Any, Callable, Literal
from waifuset import logging
from .t2i_dataset import T2IDataset
from ..utils import dataset_utils

CONDITION_IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ]
)

CNAUX_PROCESSORS = {}


class ImageConditionDataset(T2IDataset):
    condition_image_type: Literal[
        "canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
        "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
        "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
        "scribble_hed", "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
        "softedge_pidinet", "softedge_pidsafe", "dwpose"
    ] = None
    condition_image_getter: Callable = lambda self, img_md, *args, **kwargs: None
    condition_image_getter_kwargs: Dict[str, Any] = {}
    condition_image_resampling: str = 'lanczos'
    cache_condition_image: bool = False
    condition_image_cache_dir: str = None
    keep_condition_image_in_memory: bool = False

    # ControlNet++ specific
    use_random_condition_image_type: bool = False
    random_condition_image_types: List[str] = ['canny', 'depth_midas', 'lineart_anime', 'mlsd', 'normal_midas', 'scribble_hed', 'softedge_hed']

    def check_config(self):
        if not self.condition_image_getter_kwargs:
            self.condition_image_getter_kwargs = {}

        if self.condition_image_type is not None:
            if self.condition_image_getter is not None:
                self.logger.info(f"Overwrite default condition image getter with condition image type: {logging.yellow(self.condition_image_type)}")
            else:
                self.logger.info(f"Using default condition image type: {logging.yellow(self.condition_image_type)}")

            if self.condition_image_getter_kwargs and 'condition_type' in self.condition_image_getter_kwargs:
                self.logger.warning(f"Overwrite condition image getter kwargs condition_type with condition image type: {logging.yellow(self.condition_image_type)}")
            self.logger.info(f"Control image getter kwargs: {self.condition_image_getter_kwargs}")

        if self.cache_condition_image:
            if not self.condition_image_cache_dir:
                raise ValueError("Control image cache dir is not set")

        if self.condition_image_cache_dir:
            if not self.cache_condition_image:
                self.logger.warning("Control image cache dir is set but cache condition image is not enabled, condition image will not be cached")
            elif not self.condition_image_getter and not self.condition_image_type:
                self.logger.warning("Control image cache dir is set but condition image getter or type is not set, condition image will not be cached")
            else:
                self.logger.info(f"Control image cache dir: {logging.yellow(self.condition_image_cache_dir)}")
                os.makedirs(self.condition_image_cache_dir, exist_ok=True)

    def get_condition_image_cache_path(self, img_md):
        if not self.condition_image_cache_dir:
            return None
        return os.path.join(self.condition_image_cache_dir, f"{img_md['image_key']}.png")

    def open_condition_image(self, img_md) -> Image.Image:
        if self.use_random_condition_image_type:
            condition_image_type = random.choice(self.random_condition_image_types)
            condition_image = get_controlnet_aux_condition(self.get_image(img_md), condition_type=condition_image_type)
            img_md['condition_image_type'] = condition_image_type
            if condition_image is None:
                self.logger.warning(f"Failed to get condition image for random condition image type: {condition_image_type} for image {img_md.get('image_key')}")
            return condition_image
        elif self.cache_condition_image and (condition_image_cache_path := self.get_condition_image_cache_path(img_md)) is not None and os.path.exists(condition_image_cache_path) and (condition_image := Image.open(condition_image_cache_path)) is not None and condition_image.verify():
            pass
        elif (condition_image_type := self.get_condition_image_type(img_md)) is not None and (condition_image := get_controlnet_aux_condition(self.get_image(img_md), condition_type=condition_image_type)) is not None:
            if self.cache_condition_image and self.condition_image_cache_dir:
                if not os.path.exists(condition_image_cache_path):
                    condition_image.save(condition_image_cache_path)
        elif self.condition_image_getter and (condition_image := (self.condition_image_getter(self.open_image(img_md), **self.condition_image_getter_kwargs))) is not None:
            if isinstance(condition_image, Image.Image):
                pass
            elif isinstance(condition_image, np.ndarray):
                condition_image = Image.fromarray(condition_image)
            else:
                raise ValueError(f"Control image must be a PIL Image or a numpy array, got {type(condition_image)}, {condition_image}")
            if self.condition_image_cache_dir:
                if not os.path.exists(condition_image_cache_path):
                    condition_image.save(condition_image_cache_path)
        elif (condition_image := img_md.get('condition_image')) is not None:
            if isinstance(condition_image, Image.Image):
                pass
            elif isinstance(condition_image, dict):
                if (condition_image_bytes := condition_image.get('bytes')) is not None:
                    condition_image = Image.open(io.BytesIO(condition_image_bytes))
                elif (condition_image_path := condition_image.get('path')) is not None:
                    condition_image = Image.open(condition_image_path)
                else:
                    raise ValueError(f"Control image not found for {img_md.get('image_key')}, condition_image: {condition_image}")

        elif (condition_image_path := img_md.get('condition_image_path')) is not None:
            condition_image = Image.open(condition_image_path)
        else:
            self.logger.warning(f"Control image not found for {img_md.get('image_key')}")
            return None
        assert isinstance(condition_image, Image.Image), f"condition image must be a PIL Image, got {type(condition_image)}, {condition_image}"
        return condition_image

    def get_condition_image(self, img_md, type: Literal['pil', 'tensor', 'numpy'] = 'tensor') -> torch.Tensor:
        condition_image = self.open_condition_image(img_md)
        if condition_image is None:
            return None
        condition_image = condition_image.convert('RGB')
        image_size, _, bucket_size = self.get_size(img_md, update=True)

        crop_ltrb = self.get_crop_ltrb(img_md, update=True)
        condition_image = dataset_utils.resize_if_needed(condition_image, image_size, resampling=self.condition_image_resampling)
        condition_image = dataset_utils.crop_ltrb_if_needed(condition_image, crop_ltrb)
        condition_image = dataset_utils.resize_if_needed(condition_image, bucket_size, resampling=self.condition_image_resampling)
        if type == 'tensor':
            condition_image = CONDITION_IMAGE_TRANSFORMS(condition_image)
        elif type == 'numpy':
            condition_image = np.array(condition_image)
        elif type == 'pil':
            pass
        else:
            raise ValueError(f"Invalid condition image type: {type}, must be 'pil', 'tensor' or 'numpy'")
        return condition_image

    def get_condition_image_type(self, img_md) -> str:
        if 'condition_image_type' in img_md:
            return img_md['condition_image_type']
        elif self.condition_image_type is not None:
            return self.condition_image_type
        else:
            return None  # unknown

    def get_condition_image_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict[str, Any]:
        sample = dict(
            condition_images=[],
            condition_image_types=[],
        )
        for i, img_key in enumerate(batch):
            img_md = self.dataset[img_key]
            condition_image = self.get_condition_image(img_md)
            is_flipped = samples['is_flipped'][i]
            if is_flipped:
                condition_image = torch.flip(condition_image, dims=[2])
            sample["condition_images"].append(condition_image)
            condition_image_type = self.get_condition_image_type(img_md)
            sample["condition_image_types"].append(condition_image_type)

        sample["condition_images"] = torch.stack(sample["condition_images"], dim=0).to(memory_format=torch.contiguous_format).float()
        return sample


def get_controlnet_aux_condition(
    image: Image.Image,
    condition_type: Literal[
        "canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
        "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
        "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
        "scribble_hed", "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
        "softedge_pidinet", "softedge_pidsafe", "dwpose"
    ],
    **kwargs
) -> Image.Image:
    r"""
    Get the condition of an image using conditionnet_aux library.
    """
    global CNAUX_PROCESSORS
    # options are:
    # ["canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", "lineart_anime",
    #  "lineart_coarse", "lineart_realistic", "mediapipe_face", "mlsd", "normal_bae", "normal_midas",
    #  "openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand",
    #  "scribble_hed, "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe",
    #  "softedge_pidinet", "softedge_pidsafe", "dwpose"]
    if condition_type not in CNAUX_PROCESSORS:
        from controlnet_aux.processor import Processor
        CNAUX_PROCESSORS[condition_type] = Processor(condition_type, params=kwargs)
    processor = CNAUX_PROCESSORS[condition_type]
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
