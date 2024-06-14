import os
import torch
import random
import numpy as np
from PIL import Image
from typing import List, Callable, Dict, Any
from waifuset.const import IMAGE_EXTS
from waifuset.classes import DictDataset
from waifuset.tools import mapping
from .mixins.aspect_ratio_bucket_mixin import AspectRatioBucketMixin
from .mixins.cache_latents_mixin import CacheLatentsMixin
from ..utils import log_utils, dataset_utils, class_utils


class T2ITrainDataset(torch.utils.data.Dataset, AspectRatioBucketMixin, CacheLatentsMixin, class_utils.FromConfigMixin):
    metadata_files: List[str] = None
    image_dirs: List[str] = None
    dataset_name_or_path: str = None
    dataset_split: str = 'train'
    dataset_image_column: str = 'image'
    dataset_caption_column: str = 'caption'
    max_retries: int = None  # infinite retries
    hf_cache_dir: str = None
    hf_token: str = None
    batch_size: int = 1
    flip_aug: bool = False
    records_cache_dir: str = None
    dataset_info_getter: Callable = lambda self, dataset, *args, **kwargs: None
    data_weight_getter: Callable = lambda self, img_md, *args, **kwargs: 1
    caption_postprocessor: Callable = lambda self, img_md, *args, **kwargs: img_md['caption']
    description_postprocessor: Callable = lambda self, img_md, *args, **kwargs: img_md['description']

    @classmethod
    def from_config(cls, config, accelerator, **kwargs):
        return super().from_config(config, accelerator=accelerator, **kwargs)

    def setup(self):
        self.logger = log_utils.get_logger("dataset", disable=not self.accelerator.is_main_process)
        self.samplers = self.get_samplers()
        self.data = self.load_dataset()
        self.dataset_info = self.get_dataset_info()
        for img_md in self.logger.tqdm(self.data.values(), desc='prepare dataset'):
            self.load_data(img_md)
        self.logger.print(f"num_repeats: {sum(img_md.get('weight', 1) for img_md in self.data.values())}")
        self.buckets = self.make_buckets()
        if not self.arb:
            self.logger.print(f"bucket_size: {self.get_resolution()}")
        else:
            self.logger.print(f"buckets: {list(sorted(self.buckets.keys(), key=lambda x: x[0] * x[1]))}")
        self.logger.print(f"num_buckets: {len(self.buckets)}")
        self.batches = self.make_batches()
        self.logger.print(f"num_batches: {len(self.batches)}")

    def get_samplers(self):
        samplers = [sampler for sampler in dir(self) if sampler.startswith('get_') and sampler.endswith('_sample') and callable(getattr(self, sampler))]
        samplers.sort()
        samplers = [getattr(self, sampler) for sampler in samplers]
        samplers.insert(0, samplers.pop(samplers.index(self.get_t2i_sample)))
        return samplers

    def load_dataset(self) -> DictDataset:
        self.logger.print(f"loading dataset...")
        if self.metadata_files or self.image_dirs:
            dataset = self.load_local_dataset()
        elif self.dataset_name_or_path:
            if self.cache_latents is True:
                self.cache_latents = False
                self.logger.print(log_utils.yellow("Caching latents is disabled because it is not supported for Hugging Face datasets."))
            dataset = self.load_huggingface_dataset()
            if not self.arb:
                image_size = dataset[0]['image'].size

                def set_image_size(img_md):
                    img_md.setdefault('image_size', image_size)
                    return img_md

                dataset.apply_map(set_image_size)
                self.logger.print(log_utils.yellow(f"Image size is fixed to {image_size} (the first image size) in the dataset."))
        else:
            raise ValueError("Failed to load dataset. Please provide `metadata_files`, `image_dirs` or `dataset_name_or_path`.")
        self.logger.print(f"num_data: {len(dataset)}")
        self.logger.print(f"dataset: {dataset}")
        return dataset

    def load_huggingface_dataset(self) -> DictDataset:
        dataset = dataset_utils.load_huggingface_dataset(
            self.dataset_name_or_path,
            cache_dir=self.hf_cache_dir,
            split=self.dataset_split,
            primary_key='image_key',
            column_mapping=self.get_hf_dataset_column_mapping(),
            max_retries=self.max_retries,
        )
        return dataset

    def get_hf_dataset_column_mapping(self):
        return {
            self.dataset_image_column: 'image',
            self.dataset_caption_column: 'caption'
        }

    def load_local_dataset(self) -> DictDataset:
        imageset = self.load_image_dataset()
        if self.cache_latents:
            cacheset = self.load_cache_dataset()
            imageset.apply_map(mapping.redirect_columns, columns=['cache_path'], tarset=cacheset)
        return imageset

    def load_image_dataset(self) -> DictDataset:
        imageset = dataset_utils.load_local_dataset(
            self.metadata_files,
            self.image_dirs,
            tbname='metadata',
            primary_key='image_key',
            fp_key='image_path',
            exts=IMAGE_EXTS,
        )
        self.logger.print(f"num_images: {len(imageset)}")
        return imageset

    def get_dataset_info(self) -> Any:
        return self.dataset_info_getter(self)

    def make_batches(self) -> List[List[str]]:
        if self.arb:
            assert hasattr(self, 'buckets'), "You must call `make_buckets` before making batches."
            batches = []
            for img_keys in self.buckets.values():
                for i in range(0, len(img_keys), self.batch_size):
                    batch = img_keys[i:i+self.batch_size]
                    batches.append(batch)
        else:
            img_keys = list(self.data.keys())
            batches = []
            for i in range(0, len(self.data), self.batch_size):
                batch = img_keys[i:i+self.batch_size]
                batches.append(batch)
        return batches

    def load_data(self, img_md) -> Dict:
        image_size, original_size, bucket_size = self.get_size(img_md)
        weight = self.get_data_weight(img_md)
        img_md.update(
            image_size=image_size,
            original_size=original_size,
            bucket_size=bucket_size,
            weight=weight,
        )

    def get_t2i_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict:
        sample = dict(
            image_keys=[],
            images=[],
            latents=[],
            captions=[],
            is_flipped=[],
        )
        for img_key in batch:
            img_md = self.data[img_key]
            is_flipped = self.flip_aug and random.random() > 0.5
            cache = self.get_cache(img_md)
            latents = cache.get('latents')
            if latents is None:
                image = self.get_bucket_image(img_md)
                if image is None:
                    raise FileNotFoundError(f"Image and cache not found for `{img_key}`")
                if is_flipped:
                    image = torch.flip(image, dims=[2])
            else:
                if is_flipped:
                    latents = torch.flip(latents, dims=[2])  # latents.shape: [C, H, W]
                image = None
                if self.keep_cached_latents_in_memory:
                    img_md.update(cache)

            extra_kwargs = dict(is_flipped=is_flipped)
            if img_md.get('description') is not None and img_md.get('caption') is not None:
                caption = self.get_caption_during_training(img_md, **extra_kwargs) if random.random() > 0.5 else self.get_description_during_training(img_md, **extra_kwargs)
                # print(f"caption of {image_info.key}: {caption}")
            elif img_md.get('description') is not None:
                caption = self.get_description_during_training(img_md, **extra_kwargs)
            elif img_md.get('caption') is not None:
                caption = self.get_caption_during_training(img_md, **extra_kwargs)
            else:
                log_utils.warn(f"no caption or tags found for image: {img_md.key}")
                caption = ''

            sample["image_keys"].append(img_key)
            sample["images"].append(image)
            sample["latents"].append(latents)
            sample["captions"].append(caption)
            sample["is_flipped"].append(is_flipped)

        sample["images"] = torch.stack(sample["images"], dim=0).to(memory_format=torch.contiguous_format).float() if sample["images"][0] is not None else None
        sample["latents"] = torch.stack(sample["latents"], dim=0) if sample["latents"][0] is not None else None

        return sample

    def get_samples(self, batch: List[str]):
        samples = {}
        for sampler in self.samplers:
            samples.update(sampler(batch, samples))
        return samples

    def __getitem__(self, i):
        batch = self.batches[i]
        sample = self.get_samples(batch)
        return sample

    def __len__(self):
        return len(self.batches)

    def get_size(self, img_md):
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
                self.logger.print(log_utils.red(f"Failed to get image size for: {img_md}"))
                raise ValueError("Failed to get image size.")
        image_size = dataset_utils.convert_size_if_needed(image_size)

        if (original_size := img_md.get('original_size')) is None:
            original_size = image_size
        original_size = dataset_utils.convert_size_if_needed(original_size)

        if bucket_size is None and (bucket_size := img_md.get('bucket_size')) is None:
            bucket_size = self.get_bucket_size(img_md, image_size=image_size)
        bucket_size = dataset_utils.convert_size_if_needed(bucket_size)
        return image_size, original_size, bucket_size

    def open_image(self, img_md) -> Image.Image:
        if (image := img_md.get('image')) is not None:
            pass
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

    def get_bucket_image(self, img_md):
        image = self.get_image(img_md)
        if image is None:
            return None
        target_size = img_md['bucket_size']
        image, crop_ltrb = dataset_utils.crop_if_needed(image, target_size, max_ar=self.max_aspect_ratio)
        image = dataset_utils.resize_if_needed(image, target_size)
        image = dataset_utils.IMAGE_TRANSFORMS(image)
        img_md['crop_ltrb'] = crop_ltrb  # crop_ltrb: (left, top, right, bottom), set for sdxl
        return image

    def get_input_ids(self, caption, tokenizer):
        return dataset_utils.get_input_ids(caption, tokenizer, max_token_length=self.max_token_length)

    def get_caption_during_training(self, img_md, is_flipped=False):
        return self.caption_postprocessor(img_md, dataset_info=self.dataset_info, is_flipped=is_flipped)

    def get_description_during_training(self, img_md, is_flipped=False):
        return self.description_postprocessor(img_md, dataset_info=self.dataset_info, is_flipped=is_flipped)

    def get_data_weight(self, img_md):
        return self.data_weight_getter(img_md, dataset_info=self.dataset_info)

    @ staticmethod
    def collate_fn(batch):
        return batch[0]
