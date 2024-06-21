import torch
import os
import functools
import operator
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Any, Callable
from waifuset.const import IMAGE_EXTS
from ...utils import dataset_utils


class ControlNetImageMixin(object):
    control_image_source: List[Dict[str, Any]]
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

    def get_control_image_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict[str, Any]:
        sample = dict(
            control_images=[],
        )
        for i, img_key in enumerate(batch):
            img_md = self.dataset[img_key]
            control_image = self.get_control_image(img_md)
            sample["control_images"].append(control_image)
        sample["control_images"] = torch.stack(sample["control_images"], dim=0).to(memory_format=torch.contiguous_format).float()
        return sample
