import random
import torch
from PIL import Image
from .t2i_dataset import T2IDataset


class ImageDataset(T2IDataset):
    flip_aug: bool = True

    def get_basic_sample(self, batch, samples):
        sample = dict(
            image_keys=[],
            orig_images=[],
            images=[],
            is_flipped=[],
        )
        for img_key in batch:
            img_md = self.dataset[img_key]
            is_flipped = self.flip_aug and random.random() < 0.5
            image = self.get_image(img_md)
            if image is None:
                raise FileNotFoundError(f"Image not found for `{img_key}`")
            if is_flipped:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            sample['image_keys'].append(img_key)
            sample['orig_images'].append(image)
            sample['images'].append(image)
            sample['is_flipped'].append(is_flipped)
        sample["images"] = torch.stack(sample["images"], dim=0).to(memory_format=torch.contiguous_format).float()
        return sample
