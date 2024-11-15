import torch
import random
import numpy as np
from typing import List, Dict, Any
from .t2i_dataset import T2IDataset


class VAEDataset(T2IDataset):
    def check_config(self):
        super().check_config()

        if self.cache_latents:
            raise ValueError("VAEDataset does not support `cache_latents`")

    def get_basic_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict:
        sample = dict(
            image_keys=[],
            image_mds=[],
            images=[],
            orig_images=[],
            is_flipped=[],
        )
        for img_key in batch:
            img_md = self.get_img_md(img_key)
            is_flipped = self.flip_aug and random.random() > 0.5
            image = self.get_bucket_image(img_md)
            if image is None:
                raise FileNotFoundError(f"Image and cache not found for `{img_key}`")
            if is_flipped:
                image = torch.flip(image, dims=[2])

            sample["image_keys"].append(img_key)
            sample["image_mds"].append(img_md)
            sample["images"].append(image)
            sample["orig_images"].append(np.array(image))
            sample["is_flipped"].append(is_flipped)

        sample["images"] = torch.stack(sample["images"], dim=0).to(memory_format=torch.contiguous_format).float() if sample["images"][0] is not None else None

        return sample
