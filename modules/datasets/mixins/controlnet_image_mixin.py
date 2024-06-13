import torch
import numpy as np
from PIL import Image
from typing import Callable
from torchvision import transforms
from ...utils import dataset_utils


class ControlNetImageMixin(object):
    control_image_getter: Callable = lambda img_md, *args, **kwargs: None
    keep_control_image_in_memory: bool = False

    def open_control_image(self, img_md) -> Image.Image:
        if (control_image := img_md.get('control_image')) is not None:
            pass
        elif (control_image_path := img_md.get('control_image_path')) is not None:
            control_image = Image.open(control_image_path)
        elif (control_image := self.control_image_getter(img_md)) is not None:
            pass
        else:
            return None
        return control_image

    def get_control_image(self, img_md) -> torch.Tensor:
        control_image = self.open_control_image(img_md)
        if control_image is None:
            return None
        control_image = dataset_utils.convert_to_rgb(control_image)
        bucket_size = img_md.get('bucket_size')
        control_image = transforms.Compose(
            [
                transforms.Resize(bucket_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(bucket_size),
                transforms.ToTensor(),
            ]
        )(control_image)
        return control_image
