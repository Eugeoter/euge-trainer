import os
import numpy as np
import random
from PIL import Image
from typing import Callable, Dict
from ..utils import dataset_utils
from .base_dataset import BaseDataset


class ScoreDataset(BaseDataset):
    flip_aug: bool = True

    score_getter: Callable[[Dict], float] = lambda self, img_md, *args, **kwargs: None

    def open_image(self, img_md) -> Image.Image:
        if self.image_getter is not None and (image := self.image_getter(img_md)) is not None:
            assert isinstance(image, Image.Image), f"image must be an instance of PIL.Image.Image, but got {type(image)}: {image}"
        elif (image := img_md.get('image')) is not None:
            assert isinstance(image, Image.Image), f"image must be an instance of PIL.Image.Image, but got {type(image)}: {image}"
        elif os.path.exists(img_path := img_md.get('image_path', '')):
            image = Image.open(img_path)
        else:
            return None
        return image

    def get_image(self, img_md) -> Image.Image:
        image = self.open_image(img_md)
        if image is None:
            return None
        image = dataset_utils.rotate_image_straight(image)
        return image

    def get_score(self, img_md) -> int:
        if (score := self.score_getter(img_md)) is not None:
            pass
        elif (score := img_md.get('score')) is not None:
            pass
        else:
            raise ValueError(f"score not found in img_md: {img_md}")
        return score

    def get_basic_sample(self, batch, samples):
        sample = dict(
            image_keys=[],
            images=[],
            scores=[],
            is_flipped=[],
        )
        for img_key in batch:
            img_md = self.dataset[img_key]
            is_flipped = self.flip_aug and random.random() < 0.5
            image = self.get_image(img_md)
            if is_flipped:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            sample['image_keys'].append(img_key)
            sample['images'].append(image)
            sample['scores'].append(self.get_score(img_md))
            sample['is_flipped'].append(is_flipped)
        return sample
