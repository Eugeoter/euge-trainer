import os
import torch
import random
import numpy as np
import time
from pathlib import Path
from PIL import Image
from typing import List, Callable, Dict, Any

from waifuset import DictDataset
from .base_dataset import BaseDataset
from .mixins.aspect_ratio_bucket_mixin import AspectRatioBucketMixin
from .mixins.cache_latents_mixin import CacheLatentsMixin
from ..utils import dataset_utils


class T2IDataset(BaseDataset, AspectRatioBucketMixin, CacheLatentsMixin):
    flip_aug: bool = False

    caption_getter: Callable[[dict], str] = lambda self, img_md, *args, **kwargs: img_md.get('caption') or ''
    negative_caption_getter: Callable[[dict], str] = lambda self, img_md, *args, **kwargs: img_md.get('negative_caption') or ''

    image_resampling: str = 'lanczos'
    allow_crop: bool = True
    random_crop: bool = False
    do_classifier_free_guidance: bool = False

    def check_config(self):
        super().check_config()
        if self.random_crop and not self.allow_crop:
            raise ValueError(f"`random_crop` requires `allow_crop` to be True, but got {self.allow_crop}.")
        if self.random_crop and self.arb:
            self.logger.warning("`random_crop` is enabled but `arb` is True. This will cause unexpected behavior.")

    def get_setups(self):
        return [
            self._setup_basic,
            self._setup_dataset,
            self._setup_pre_dataset_hook,
            self._setup_data,
            self._setup_post_dataset_hook,
            self._setup_buckets,
            self._setup_batches,
        ]

    def _setup_buckets(self):
        self.buckets = self.make_buckets(self.dataset)

        if not self.arb:
            self.logger.print(f"resolution: {self.get_resolution()}")
        else:
            bucket_keys = list(sorted(self.buckets.keys(), key=lambda x: x[0] * x[1]))
            self.logger.print(f"buckets: {bucket_keys[0]} ~ {bucket_keys[-1]}")
            self.logger.print(f"number of buckets: {len(self.buckets)}")

    def get_samplers(self):
        samplers = super().get_samplers()
        samplers.insert(0, samplers.pop(samplers.index(self.get_basic_sample)))
        return samplers

    def get_basic_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict:
        sample = dict(
            image_keys=[],
            image_mds=[],
            images=[],
            latents=[],
            captions=[],
            negative_captions=[] if self.do_classifier_free_guidance else None,
            is_flipped=[],
        )
        for img_key in batch:
            img_md = self.get_img_md(img_key)
            is_flipped = self.flip_aug and random.random() > 0.5
            if (latents := (cache := self.get_cache(img_md, update=self.keep_cached_latents_in_memory)).get('latents')) is not None:
                if is_flipped:
                    latents = torch.flip(latents, dims=[2])  # latents.shape: [C, H, W]
                image = None
                if self.keep_cached_latents_in_memory:
                    img_md.update(cache)
            else:
                image = self.get_bucket_image(img_md)
                if image is None:
                    raise FileNotFoundError(f"Image and cache not found for `{img_key}`")
                if is_flipped:
                    image = torch.flip(image, dims=[2])

            caption = self.get_caption(img_md, is_flipped=is_flipped)
            if self.do_classifier_free_guidance:
                negative_caption = self.get_negative_caption(img_md, is_flipped=is_flipped)

            sample["image_keys"].append(img_key)
            sample["image_mds"].append(img_md)
            sample["images"].append(image)
            sample["latents"].append(latents)
            sample["captions"].append(caption)
            sample["is_flipped"].append(is_flipped)
            if self.do_classifier_free_guidance:
                sample["negative_captions"].append(negative_caption)

        sample["images"] = torch.stack(sample["images"], dim=0).to(memory_format=torch.contiguous_format).float() if sample["images"][0] is not None else None
        sample["latents"] = torch.stack(sample["latents"], dim=0) if sample["latents"][0] is not None else None

        return sample

    def get_fp_keys(self):
        if isinstance(self.dataset_source, str):
            self.dataset_source = [self.dataset_source]
        self.dataset_source = [dict(name_or_path=source) if isinstance(source, str) else source for source in self.dataset_source]
        fp_keys = []
        for ds_src in self.dataset_source:
            if (fp_key := ds_src.get('fp_key')) is not None:
                fp_keys.append(fp_key)
        fp_keys.append('cache_path')
        return fp_keys

    def load_data(self, img_md) -> Dict:
        img_md = self.get_preprocessed_img_md(img_md)
        extra_kwargs = {}
        if not self.cache_only:
            weight = self.get_data_weight(img_md)
            if weight == 0:
                img_md.update(drop=True)
                return
            extra_kwargs.update(weight=weight)
        if self.arb:
            image_size, original_size, bucket_size = self.get_size(img_md)
            if self.max_width is not None and image_size[0] > self.max_width:
                self.logger.warning(f"drop image: {img_md['image_key']} due to image size {image_size} > max width {self.max_width}")
                img_md.update(drop=True)
                return
            elif self.max_height is not None and image_size[1] > self.max_height:
                self.logger.warning(f"drop image: {img_md['image_key']} due to image size {image_size} > max height {self.max_height}")
                img_md.update(drop=True)
                return
            elif self.max_area is not None and bucket_size[0] * bucket_size[1] > self.max_area:
                self.logger.warning(f"drop image: {img_md['image_key']} due to bucket size {bucket_size} > resolution {self.get_resolution()}")
                img_md.update(drop=True)
                return
            extra_kwargs.update(
                image_size=image_size,
                original_size=original_size,
                bucket_size=bucket_size,
            )
        img_md.update(**extra_kwargs)

    def make_batches(self, dataset) -> List[List[str]]:
        if self.arb:
            assert hasattr(self, 'buckets'), "You must call `make_buckets` before making batches."
            if self.batch_size == 1:
                batches = [[img_key] for img_key in dataset.keys()]
            else:
                batches = []
                for img_keys in self.logger.tqdm(self.buckets.values(), desc='make batches'):
                    for i in range(0, len(img_keys), self.batch_size):
                        batch = img_keys[i:i+self.batch_size]
                        batches.append(batch)
        else:
            batches = super().make_batches(dataset)
        return batches

    def get_latents(self, img_md):
        return None if (cache := self.get_cache(img_md, update=self.keep_cached_latents_in_memory)) is None else cache.get('latents')

    def get_size(self, img_md, update=False):
        image_size, original_size, bucket_size = None, None, None
        if (image_size := img_md.get('image_size')) is None:
            if os.path.exists(img_path := img_md.get('image_path', '')):
                image_size = dataset_utils.get_image_size(img_path)
            elif (image := self.open_image(img_md)) is not None:
                image_size = image.size
            elif (latents_size := self.get_latents_size(img_md)) is not None:
                image_size = (latents_size[0] * 8, latents_size[1] * 8)
                bucket_size = image_size  # precomputed latents
            else:
                self.logger.error(f"Failed to get image size for: {img_md}")
                raise ValueError("Failed to get image size.")
        image_size = dataset_utils.convert_size_if_needed(image_size)

        if (original_size := img_md.get('original_size')) is None:
            original_size = image_size
        original_size = dataset_utils.convert_size_if_needed(original_size)

        if bucket_size is None and (bucket_size := img_md.get('bucket_size')) is None:
            bucket_size = self.get_bucket_size(img_md, image_size=image_size)
        bucket_size = dataset_utils.convert_size_if_needed(bucket_size)

        if update:
            img_md.update(
                image_size=image_size,
                original_size=original_size,
                bucket_size=bucket_size,
            )
        return image_size, original_size, bucket_size

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

    def get_image(self, img_md):
        image = self.open_image(img_md)
        if image is None:
            return None
        image = dataset_utils.convert_to_rgb(image)
        image = dataset_utils.rotate_image_straight(image)
        image = np.array(image, np.uint8)  # (H, W, C)
        return image

    def get_bucket_image(self, img_md) -> torch.Tensor:
        image = self.get_image(img_md)
        if image is None:
            return None
        _, _, bucket_size = self.get_size(img_md, update=True)
        crop_ltrb = self.get_crop_ltrb(img_md, update=True)
        image = dataset_utils.crop_ltrb_if_needed(image, crop_ltrb)
        image = dataset_utils.resize_if_needed(image, bucket_size, resampling=self.image_resampling)
        image = dataset_utils.IMAGE_TRANSFORMS(image)
        img_md['crop_ltrb'] = crop_ltrb  # crop_ltrb: (left, top, right, bottom), set for sdxl
        return image

    def get_crop_ltrb(self, img_md, update=True):
        if img_md.get('crop_ltrb') is not None:
            return img_md['crop_ltrb']

        image_size = img_md['image_size']
        if not self.allow_crop:
            crop_ltrb = (0, 0, image_size[0], image_size[1])
        elif self.random_crop:
            bucket_size = img_md['bucket_size']
            left = random.randint(0, image_size[0] - bucket_size[0]) if image_size[0] > bucket_size[0] else 0
            top = random.randint(0, image_size[1] - bucket_size[1]) if image_size[1] > bucket_size[1] else 0
            right = left + bucket_size[0] if image_size[0] > bucket_size[0] else image_size[0]
            bottom = top + bucket_size[1] if image_size[1] > bucket_size[1] else image_size[1]
            crop_ltrb = (left, top, right, bottom)
        else:
            bucket_size = img_md['bucket_size']
            max_ar = self.max_aspect_ratio
            img_w, img_h = image_size
            tar_w, tar_h = bucket_size

            ar_image = img_w / img_h
            ar_target = tar_w / tar_h

            if max_ar is not None and dataset_utils.aspect_ratio_diff(image_size, bucket_size) > max_ar:
                if ar_image < ar_target:
                    new_height = img_w / ar_target * max_ar
                    new_width = img_w
                else:
                    new_width = img_h * ar_target / max_ar
                    new_height = img_h

                left = max(0, int((img_w - new_width) / 2))
                top = max(0, int((img_h - new_height) / 2))
                right = int(left + new_width)
                bottom = int(top + new_height)
                crop_ltrb = (left, top, right, bottom)
            else:
                crop_ltrb = (0, 0, img_w, img_h)
        if update:
            img_md['crop_ltrb'] = crop_ltrb
        return crop_ltrb

    def get_input_ids(self, caption, tokenizer):
        return dataset_utils.get_input_ids(caption, tokenizer, max_token_length=self.max_token_length)

    def get_caption(self, img_md, is_flipped=False):
        return self.caption_getter(img_md, dataset_hook=self.dataset_hook, is_flipped=is_flipped)

    def get_negative_caption(self, img_md, is_flipped=False):
        return self.negative_caption_getter(img_md, dataset_hook=self.dataset_hook, is_flipped=is_flipped)
