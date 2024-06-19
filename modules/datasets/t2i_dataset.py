import os
import torch
import random
import numpy as np
import functools
import operator
from ml_collections import ConfigDict
from PIL import Image
from typing import List, Callable, Dict, Any, Union
from waifuset.const import IMAGE_EXTS
from waifuset.classes import DictDataset
from waifuset.tools import mapping
from .mixins.aspect_ratio_bucket_mixin import AspectRatioBucketMixin
from .mixins.cache_latents_mixin import CacheLatentsMixin
from ..utils import log_utils, dataset_utils, class_utils


class T2ITrainDataset(torch.utils.data.Dataset, AspectRatioBucketMixin, CacheLatentsMixin, class_utils.FromConfigMixin):
    dataset_source: Union[str, List[str]]
    metadata_source: Union[str, List[str]]
    max_retries: int = None  # infinite retries
    hf_cache_dir: str = None
    hf_token: str = None
    batch_size: int = 1
    flip_aug: bool = False
    records_cache_dir: str = None
    max_dataset_n_workers: int = 1
    data_preprocessor: Callable = lambda self, img_md, *args, **kwargs: img_md
    dataset_info_getter: Callable = lambda self, dataset, *args, **kwargs: None
    data_weight_getter: Callable = lambda self, img_md, *args, **kwargs: 1
    caption_getter: Callable = lambda self, img_md, *args, **kwargs: img_md.get('caption') or ''

    @classmethod
    def from_config(cls, config, accelerator, **kwargs):
        return super().from_config(config, accelerator=accelerator, **kwargs)

    def setup(self):
        self.logger = log_utils.get_logger("dataset", disable=not self.accelerator.is_main_process)
        self.samplers = self.get_samplers()
        self.dataset = self.load_dataset()
        self.dataset_info = self.get_dataset_info()

        if self.max_dataset_n_workers <= 1:
            for img_md in self.logger.tqdm(self.dataset.values(), desc='prepare dataset'):
                self.load_data(img_md)
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_dataset_n_workers) as executor:
                futures = {executor.submit(self.load_data, img_md): img_md for img_md in self.dataset.values()}
                for future in concurrent.futures.as_completed(futures):
                    future.result()

        # filter out data with weight 0
        for img_key, img_md in list(self.dataset.items()):
            if img_md.get('weight', 1) <= 0:
                del self.dataset[img_key]
        self.logger.print(f"number of data (after filtering): {len(self.dataset)}")
        self.logger.print(f"number of repeats: {sum(img_md.get('weight', 1) for img_md in self.dataset.values())}")
        self.buckets = self.make_buckets()
        if not self.arb:
            self.logger.print(f"resolution: {self.get_resolution()}")
        else:
            self.logger.print(f"buckets: {list(sorted(self.buckets.keys(), key=lambda x: x[0] * x[1]))}")
        self.logger.print(f"number of buckets: {len(self.buckets)}")
        self.batches = self.make_batches()
        self.logger.print(f"number pf batches: {len(self.batches)}")

    def get_samplers(self):
        samplers = [sampler for sampler in dir(self) if sampler.startswith('get_') and sampler.endswith('_sample') and callable(getattr(self, sampler))]
        samplers.sort()
        samplers = [getattr(self, sampler) for sampler in samplers]
        samplers.insert(0, samplers.pop(samplers.index(self.get_t2i_sample)))
        return samplers

    def load_dataset(self) -> DictDataset:
        self.logger.print(f"loading dataset...")
        datasets = []
        dataset_loaders = [
            dataset_loader for dataset_loader in dir(self)
            if dataset_loader.startswith('load_') and dataset_loader.endswith('_dataset') and callable(getattr(self, dataset_loader)) and dataset_loader != 'load_dataset'
        ]

        # load datasets
        with self.logger.tqdm(total=len(dataset_loaders), desc='load datasets') as pbar:
            msgs = []
            for dataset_loader in dataset_loaders:
                dataset_type = dataset_loader[5:-8]
                pbar.set_postfix(dataset=dataset_type)
                ds = getattr(self, dataset_loader)()
                msgs.append(f"number of rows in {dataset_type} dataset: {len(ds)}")
                datasets.append(ds)
                pbar.update()
            for msg in msgs:
                self.logger.print(msg)

        # merge datasets into metaset
        metaset: DictDataset = self.load_metadata()  # load metadata
        metaset = metaset.sample(n=100)
        self.logger.print(f"number of rows in metadata: {len(metaset)}")
        for img_key, old_img_md in self.logger.tqdm(metaset.items(), desc='merge datasets to metadata'):
            for ds in datasets:
                if (new_img_md := ds.get(img_key)) is not None:
                    old_img_md.update(new_img_md)
                    if issubclass(new_img_md.__class__, old_img_md.__class__):
                        new_img_md.update(old_img_md)
                        metaset[img_key] = new_img_md
        metaset.apply_map(mapping.as_posix_path, columns=('image_path', 'cache_path'))
        metaset = metaset.subset(self.get_data_existence)
        self.logger.print(f"dataset: {metaset}")
        return metaset

    def get_data_existence(self, img_md):
        return isinstance(img_md, dataset_utils.HuggingFaceData) or os.path.exists(img_md.get('image_path', '')) or os.path.exists(img_md.get('cache_path', ''))

    def load_image_dataset(self) -> DictDataset:
        """
        Load image dataset from `dataset_source`.
        """
        if isinstance(self.dataset_source, str):
            self.dataset_source = [self.dataset_source]
        self.dataset_source = [dict(name_or_path=source) if isinstance(source, str) else source for source in self.dataset_source]
        imagesets = []
        for ds_src in self.dataset_source:
            name_or_path = ds_src.get('name_or_path')
            if os.path.exists(name_or_path):
                imageset = dataset_utils.load_image_directory_dataset(
                    image_directory=name_or_path,
                    fp_key=ds_src.get('fp_key', 'image_path'),
                    recur=ds_src.get('recur', True),
                    exts=ds_src.get('exts', IMAGE_EXTS),
                )
            else:
                imageset = dataset_utils.load_huggingface_dataset(
                    name_or_path=name_or_path,
                    cache_dir=ds_src.get('cache_dir', self.hf_cache_dir),
                    split=ds_src.get('split', 'train'),
                    primary_key='image_key',
                    column_mapping=ds_src.get('column_mapping', {k: 'image' for k in ('image', 'png', 'jpg', 'jpeg', 'webp', 'jfif')}),
                    hf_token=ds_src.get('hf_token', self.hf_token),
                    max_retries=ds_src.get('max_retries', self.max_retries),
                )
            imagesets.append(imageset)
        imageset = functools.reduce(operator.add, imagesets)
        return imageset

    def load_metadata(self):
        self.logger.print(f"loading metadata...")
        if isinstance(self.metadata_source, str):
            self.metadata_source = [self.metadata_source]
        self.metadata_source = [dict(name_or_path=source) if isinstance(source, str) else source for source in self.metadata_source]
        metasets = []
        for md_src in self.metadata_source:
            md_file = md_src.get('name_or_path')
            metaset = dataset_utils.load_metadata_dataset(
                md_file,
                fp_key=md_src.get('fp_key', 'image_path'),
                recur=md_src.get('recur', True),
                exts=md_src.get('exts', IMAGE_EXTS),
                tbname=md_src.get('tbname', 'metadata'),
                primary_key='image_key',
            )
            metasets.append(metaset)
        metaset = functools.reduce(lambda x, y: x + y, metasets)
        return metaset

    def get_dataset_info(self) -> Any:
        return self.dataset_info_getter(self)

    def get_preprocessed_img_md(self, img_md) -> Dict:
        return self.data_preprocessor(img_md, dataset_info=self.dataset_info)

    def make_batches(self) -> List[List[str]]:
        if self.arb:
            assert hasattr(self, 'buckets'), "You must call `make_buckets` before making batches."
            batches = []
            for img_keys in self.buckets.values():
                for i in range(0, len(img_keys), self.batch_size):
                    batch = img_keys[i:i+self.batch_size]
                    batches.append(batch)
        else:
            img_keys = list(self.dataset.keys())
            batches = []
            for i in range(0, len(self.dataset), self.batch_size):
                batch = img_keys[i:i+self.batch_size]
                batches.append(batch)
        return batches

    def load_data(self, img_md) -> Dict:
        img_md = self.get_preprocessed_img_md(img_md)
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
            img_md = self.dataset[img_key]
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

            caption = self.get_caption(img_md, is_flipped=is_flipped)

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
        crop_ltrb = self.get_crop_ltrb(img_md)
        image = dataset_utils.crop_ltrb_if_needed(image, crop_ltrb)
        image = dataset_utils.resize_if_needed(image, img_md['bucket_size'])
        image = dataset_utils.IMAGE_TRANSFORMS(image)
        img_md['crop_ltrb'] = crop_ltrb  # crop_ltrb: (left, top, right, bottom), set for sdxl
        return image

    def get_crop_ltrb(self, img_md):
        image_size = img_md['image_size']
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
        return crop_ltrb

    def get_input_ids(self, caption, tokenizer):
        return dataset_utils.get_input_ids(caption, tokenizer, max_token_length=self.max_token_length)

    def get_caption(self, img_md, is_flipped=False):
        return self.caption_getter(img_md, dataset_info=self.dataset_info, is_flipped=is_flipped)

    def get_data_weight(self, img_md):
        return self.data_weight_getter(img_md, dataset_info=self.dataset_info)

    @ staticmethod
    def collate_fn(batch):
        return batch[0]
