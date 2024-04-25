import os
import json
import torch
import math
import random
import numpy as np
import cv2
import aiofiles
import asyncio
from pathlib import Path
from PIL import Image, ExifTags
from typing import List, Tuple, Optional, Union
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, wait
from . import log_utils

SDXL_BUCKET_RESOS = [
    (512, 1856), (512, 1920), (512, 1984), (512, 2048),
    (576, 1664), (576, 1728), (576, 1792), (640, 1536),
    (640, 1600), (704, 1344), (704, 1408), (704, 1472),
    (768, 1280), (768, 1344), (832, 1152), (832, 1216),
    (896, 1088), (896, 1152), (960, 1024), (960, 1088),
    (1024, 960), (1024, 1024), (1088, 896), (1088, 960),
    (1152, 832), (1152, 896), (1216, 832), (1280, 768),
    (1344, 704), (1344, 768), (1408, 704), (1472, 704),
    (1536, 640), (1600, 640), (1664, 576), (1728, 576),
    (1792, 576), (1856, 512), (1920, 512), (1984, 512), (2048, 512)
]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]
IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


class ImageInfo:
    def __init__(
        self,
        key: str,
        image_path,
        caption: str = None,
        description: str = None,
        image_size=None,  # (w, h)
        original_size=None,  # (w, h)
        crop_ltrb=None,
        latent_size=None,  # (w, h)
        bucket_size=None,
        num_repeats=1,
        npz_path=None,
        latents=None,
        latents_flipped=None,
        metadata=None,
    ):
        self.key = key
        self.caption = caption
        self.description = description
        self.image_path = image_path
        self.image_size = image_size
        self.original_size = original_size
        self.crop_ltrb = crop_ltrb
        self.latent_size = latent_size
        self.bucket_size = bucket_size
        self.num_repeats = num_repeats
        self.npz_path = npz_path
        self.latents = latents
        self.latents_flipped = latents_flipped
        self.metadata = metadata

    def dict(self):
        return dict(
            key=self.key,
            caption=self.caption,
            description=self.description,
            image_path=self.image_path,
            image_size=self.image_size,
            original_size=self.original_size,
            latent_size=self.latent_size,
            bucket_size=self.bucket_size,
            num_repeats=self.num_repeats,
            npz_path=self.npz_path,
            latents=self.latents,
            latents_flipped=self.latents_flipped,
            metadata=self.metadata,
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        tokenizer1,
        tokenizer2,
        latents_dtype=torch.float32,
        predefined_bucket_resos=SDXL_BUCKET_RESOS,
        is_main_process=False,
        num_processes=1,
        process_idx=0,
        cache_only=False,
    ):
        self.image_dirs = [Path(image_dir).absolute() for image_dir in config.image_dirs]
        self.metadata_files = [Path(metadata_file).absolute() for metadata_file in config.metadata_files]
        self.records_dir = Path(config.records_cache_dir).absolute() if config.records_cache_dir else None
        self.batch_size = config.batch_size
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2

        self.latents_dtype = latents_dtype
        self.max_token_length = config.max_token_length
        self.predefined_bucket_resos = predefined_bucket_resos

        self.num_repeats_getter = config.num_repeats_getter or (lambda *args, **kwargs: 1)
        self.caption_processor = config.caption_processor or (lambda img_info, *args, **kwargs: img_info.caption)
        self.description_processor = config.description_processor or (lambda img_info, *args, **kwargs: img_info.description)

        self.resolution = config.resolution
        self.bucket_reso_step = config.bucket_reso_step
        self.flip_aug = config.flip_aug

        self.check_cache_validity = config.check_cache_validity
        self.keep_cached_latents_in_memory = config.keep_cached_latents_in_memory

        self.max_workers = min(config.max_dataset_n_workers, os.cpu_count() - 1)
        self.is_main_process = is_main_process
        self.num_processes = num_processes
        self.process_idx = process_idx
        self.cache_only = cache_only

        self.image_data = {}
        self.buckets = {}
        self.batches = []

        self.logger = log_utils.get_logger("dataset" if not self.cache_only else "cache", disable=not is_main_process)

        # for accelerating searching
        stem2files = {}
        img_exts = set(IMAGE_EXTENSIONS)
        exts = img_exts | {'.npz'}
        file_cnt = 0
        for img_dir in self.logger.tqdm(self.image_dirs, desc=f"indexing image files"):
            imfiles = listdir(img_dir, return_path=True, return_type=Path, recur=True, exts=exts)
            file_cnt += len(imfiles)
            for p in imfiles:
                stem2files.setdefault(p.stem, []).append(p)
        self.logger.print(f"num_files: {log_utils.yellow(file_cnt)} | num_stems: {log_utils.yellow(len(stem2files))}")

        self.metadata = {}
        if self.metadata_files:
            self.logger.print(f"load from metadata files:\n  " + '\n  '.join([log_utils.yellow(str(metadata_file)) for metadata_file in self.metadata_files]))
            for metadata_file in self.metadata_files:
                if metadata_file.is_file():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    self.metadata.update(metadata)
                else:
                    raise FileNotFoundError(f"metadata file not found: {metadata_file}")
        else:
            self.logger.print(f"load from image dirs:\n  " + '\n  '.join([log_utils.yellow(str(img_dir)) for img_dir in self.image_dirs]))
            for stem, files in stem2files.items():
                files = [file for file in files if file.suffix in img_exts]
                assert len(files) == 1, f"multiple image files found for stem `{stem}`: {files}"
                img_md = empty_metadata()
                img_path = files[0]
                img_md['image_path'] = str(img_path)
                cap_path = img_path.with_suffix('.txt')
                if cap_path.is_file():
                    with open(cap_path, 'r') as f:
                        caption = f.read().strip()
                    img_md.update(caption2metadata(caption))
                self.metadata[stem] = img_md

        self.num_train_images = 0
        self.num_train_repeats = 0

        if self.cache_only:
            self.logger.print(f"run in `cache_only` mode. `num_repeats` will be fixed to 1.")

        # load img_size_record
        if self.records_dir:
            img_size_log_path = self.records_dir / "image_size.json"
            n_rep_log_path = self.records_dir / "num_repeats.json"
            counter_log_path = self.records_dir / "counter.json"
            img_size_record = {}
            if img_size_log_path.is_file():
                try:
                    with open(img_size_log_path) as f:
                        img_size_record = json.load(f)
                    self.logger.print(f"applied existing `image size` record: {img_size_log_path} | size: {log_utils.yellow(len(img_size_record))}")
                except json.JSONDecodeError:
                    img_size_record = {}
                    self.logger.print(log_utils.red(f"failed to load `image size` record: {img_size_log_path}"))
            else:
                self.logger.print(log_utils.yellow(f"no existing `image size` record found: {img_size_log_path}"))

            num_repeats_record = {}

        logs = {}
        log_counter = {
            'caption': 0,
            'description': 0,
            'missing': 0,
        }

        pbar = self.logger.tqdm(total=len(self.metadata), desc=f"searching dataset")

        @log_utils.track_tqdm(pbar)
        def search_data(img_key, img_md):
            # 1. search image or cache file
            # 1.1 search from metadata
            if img_path := img_md.get('image_path'):
                if '\\' in img_path:
                    img_md['image_path'] = img_path = img_path.replace('\\', '/')
                img_ext = os.path.splitext(img_path)[-1]
                if os.path.exists(img_path):
                    img_path = Path(img_path)
                    npz_path = img_path.with_suffix('.npz')
                    if not npz_path.exists():
                        npz_path = None
                else:
                    img_path = None
                    npz_path = None
            # 1.2 search from file table
            if self.image_dirs and img_key in stem2files:
                if img_ext:
                    img_path = search_file(stem2files[img_key], exts=(img_ext,))
                if img_path is None:
                    img_path = search_file(stem2files[img_key], exts=img_exts)
                npz_path = search_file(stem2files[img_key], exts=('.npz',))

            img_md['missing'] = img_path is None and npz_path is None
            # img_md['image_path'] = img_path
            img_md['npz_path'] = npz_path

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(search_data, img_key, img_md) for img_key, img_md in self.metadata.items()]
                wait(futures)
        else:
            for img_key, img_md in self.metadata.items():
                search_data(img_key, img_md)

        for img_key in list(self.metadata.keys()):
            img_md = self.metadata[img_key]
            if img_md.get('missing'):
                log_counter['missing'] += 1
                del self.metadata[img_key]

        pbar.close()
        self.counter = count_metadata(self.metadata)
        pbar = self.logger.tqdm(total=len(self.metadata), desc=f"loading dataset")

        @log_utils.track_tqdm(pbar)
        def load_data(img_key, img_md):
            # 1. get image_path and npz_path
            img_path = img_md.get('image_path')
            npz_path = img_md.get('npz_path')

            # 2. get num_repeats
            num_repeats = self.num_repeats_getter(img_key, img_md, counter=self.counter)
            if self.cache_only:
                num_repeats = min(num_repeats, 1)
            if num_repeats == 0:  # drop
                return
            img_md['num_repeats'] = num_repeats

            # 3. get caption & description
            caption = img_md.get('caption') or img_md.get('tags')  # tag caption
            desc = img_md.get('description') or img_md.get('nl_caption')  # natural language caption
            log_counter['caption'] += 1 if caption else 0
            log_counter['description'] += 1 if desc else 0

            # 4. get image size
            image_size = None
            latents_size = None
            bucket_reso = None
            if image_size:  # if image_size is provided, pass
                pass
            elif self.records_dir and img_key in img_size_record:  # if image_size is cached, use it
                image_size = tuple(img_size_record[img_key])
            elif img_path is not None and (not npz_path or (npz_path and self.check_cache_validity)):  # if image_size is not provided, try to read from image file
                image_size = get_image_size(img_path)
            else:
                latents_size = get_latent_image_size(npz_path)
                if latents_size is None:
                    raise RuntimeError(f"failed to read image size: `{img_key}`. Please check if the image file exists or the cached latent is valid.")
                image_size = (latents_size[0] * 8, latents_size[1] * 8)
                bucket_reso = image_size  # ! directly use latents_size as bucket_reso

            original_size = img_md.get('original_size', None)

            # 5. record and log
            if self.records_dir:
                num_repeats_record[img_key] = num_repeats
                img_size_record[img_key] = image_size

            artist = img_md.get('artist', None)
            if artist is not None and artist not in logs:
                logs[artist] = num_repeats

            img_info = ImageInfo(
                key=img_key,
                caption=caption,
                description=desc,
                image_path=img_path,
                npz_path=npz_path,
                image_size=image_size,
                original_size=original_size,
                latent_size=latents_size,
                bucket_size=bucket_reso,
                latents=None,
                latents_flipped=None,
                num_repeats=num_repeats,
                metadata=img_md,
            )

            self.register_image_info(img_info)

            self.num_train_images += 1
            self.num_train_repeats += num_repeats

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(load_data, img_key, img_md) for img_key, img_md in self.metadata.items()]
                wait(futures)
        else:
            for img_key, img_md in self.metadata.items():
                load_data(img_key, img_md)

        pbar.close()

        if self.is_main_process:  # log and record
            self.logger.print(f"num_train_images: {log_utils.yellow(self.num_train_images)} | num_train_repeats: {log_utils.yellow(self.num_train_repeats)}")
            self.logger.print(f"quality counter:", ' | '.join([f"{k}: {log_utils.yellow(v)}" for k, v in self.counter['quality'].items()]))
            self.logger.print(
                f"num_captioned: {log_utils.yellow(log_counter['caption'])}/{len(self.image_data)} | num_described: {log_utils.yellow(log_counter['description'])}/{len(self.image_data)} | num_missing: {log_utils.yellow(log_counter['missing'])}/{len(self.metadata)}")
            if self.records_dir:
                n_rep_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(n_rep_log_path, 'w') as f:
                    json.dump(num_repeats_record, f)
                img_size_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(img_size_log_path, 'w') as f:
                    json.dump(img_size_record, f)
                self.logger.print(f"records to: `{log_utils.yellow(self.records_dir)}`")
                counter_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(counter_log_path, 'w') as f:
                    json.dump(self.counter, f)

        # count metadata again
        self.counter = count_metadata(self.metadata)

        self.make_buckets()
        self.make_batches()

    def log(self, *args, prefix='dataset', **kwargs):
        if self.is_main_process:
            print(log_utils.blue('['+prefix+']'), *args, **kwargs)

    def register_image_info(self, image_info):
        self.image_data[image_info.key] = image_info

    def process_caption(self, image_info: ImageInfo, flip_aug=False):
        return self.caption_processor(image_info, counter=self.counter, flip_aug=flip_aug)

    def process_description(self, image_info: ImageInfo, flip_aug=False):
        return self.description_processor(image_info, counter=self.counter, flip_aug=flip_aug)

    def get_input_ids(self, caption, tokenizer):
        return get_input_ids(caption, tokenizer, max_token_length=self.max_token_length)

    def shuffle_buckets(self):
        bucket_keys = list(self.buckets.keys())
        random.shuffle(bucket_keys)
        self.buckets = {k: self.buckets[k] for k in bucket_keys}
        for bucket in self.buckets.values():
            random.shuffle(bucket)

    def make_buckets(self):
        for img_info in self.logger.tqdm(self.image_data.values(), desc=f"making buckets", disable=not self.is_main_process):
            img_info: ImageInfo
            if img_info.bucket_size is not None:
                pass
            elif img_info.latent_size is not None or img_info.latents is not None:
                if img_info.latents is not None:  # latents.shape: [C, H, W]
                    bucket_reso = (img_info.latents.shape[-1] * 8, img_info.latents.shape[-2] * 8)
                else:
                    bucket_reso = (img_info.latent_size[0] * 8, img_info.latent_size[1] * 8)
                img_info.bucket_size = bucket_reso
                # assert expected_bucket_reso == img_info.bucket_size, f"latent size and bucket reso of `{img_info.key}` mismatch: excepted bucket reso to be {expected_bucket_reso}, but got {img_info.bucket_size}"
            else:  # make from image file
                bucket_reso = get_bucket_reso(img_info.image_size, buckets=self.predefined_bucket_resos,
                                              max_resolution=self.resolution, divisible=self.bucket_reso_step)
                img_info.bucket_size = bucket_reso

            assert img_info.bucket_size[0] % self.bucket_reso_step == 0 and img_info.bucket_size[1] % self.bucket_reso_step == 0, \
                f"bucket reso must be divisible by {self.bucket_reso_step}: {img_info.bucket_size}"

            if img_info.bucket_size not in self.buckets:
                self.buckets[img_info.bucket_size] = []
            self.buckets[img_info.bucket_size].extend([img_info] * img_info.num_repeats)

        self.shuffle_buckets()

    def debug_buckets(self):
        for i, bucket_reso in enumerate(self.buckets):
            bucket = self.buckets[bucket_reso]
            for j, img_info in enumerate(bucket):
                self.logger.print(f"  [{j}]: {img_info.key} | {img_info.image_size} -> {img_info.bucket_size}")

    def make_batches(self):
        self.batches = []
        for bucket in self.logger.tqdm(self.buckets.values(), desc=f"making batches", disable=not self.is_main_process):
            for i in range(0, len(bucket), self.batch_size):
                self.batches.append(bucket[i:i+self.batch_size])

    def cache_latents(self, vae, accelerator, vae_batch_size=1, cache_to_disk=False, check_validity=False, empty_cache=False, async_cache=False):
        if self.cache_only and not cache_to_disk:
            cache_to_disk = True
            self.logger.print(log_utils.yellow("cache_only is enabled. cache_to_disk is forced to be True."))
        image_infos = list(self.image_data.values())
        image_infos.sort(key=lambda info: info.bucket_size[0] * info.bucket_size[1])
        batches = []  # uncached batches
        batch = []
        pbar = self.logger.tqdm(total=len(image_infos), desc=f"checking latents")
        pbar_logs = {
            'invalid': 0,
            'miss': 0,
        }
        pbar.set_postfix(pbar_logs)
        for image_info in image_infos:
            if image_info.latents is not None and (not self.flip_aug or image_info.latents_flipped is not None):
                pbar.update(1)
                continue

            if image_info.npz_path:  # if npz file exists
                if not check_validity:
                    pbar.update(1)
                    continue
                latents, latents_flipped, orig_size, crop_ltrb = load_latents_from_disk(image_info.npz_path, flip_aug=self.flip_aug, dtype=self.latents_dtype, is_main_process=self.is_main_process)
                if orig_size is not None:
                    image_info.original_size = orig_size
                if crop_ltrb is not None:
                    image_info.crop_ltrb = crop_ltrb

                image_info.latents = latents
                image_info.latents_flipped = latents_flipped
                # print(f"check latents: {image_info.key} | {image_info.npz_path}")
                lat_val = check_cached_latents(image_info, latents)
                lat_flip_val = (not self.flip_aug or check_cached_latents(image_info, latents_flipped))
                if lat_val and lat_flip_val:  # if latents is valid
                    pbar.update(1)
                    continue
                else:
                    pbar_logs['miss'] += 1
                    pbar.set_postfix(pbar_logs)
                    if not lat_val:
                        image_info.latents = None
                    if not lat_flip_val:
                        image_info.latents_flipped = None
            else:
                pbar_logs['invalid'] += 1
                pbar.set_postfix(pbar_logs)

            if len(batch) > 0 and batch[-1].bucket_size != image_info.bucket_size:
                batches.append(batch)
                batch = []

            batch.append(image_info)

            # if number of data in batch is enough, flush the batch
            if len(batch) >= vae_batch_size:
                batches.append(batch)
                batch = []
            pbar.update(1)

        pbar.close()

        if len(batch) > 0:
            batches.append(batch)

        total_num_batches = len(batches)

        if total_num_batches == 0:
            self.logger.print(log_utils.green("all latents are cached"))
            return

        if self.num_processes > 1:
            batches = batches[self.process_idx::self.num_processes]  # split batches into processes
            self.logger.print(f"process {self.process_idx+1}/{self.num_processes} | num_uncached_batches: {len(batches)}", disable=False)

        self.logger.print(f"total: {len(batches)} x {vae_batch_size} x {self.num_processes} ≈ {total_num_batches*vae_batch_size} (difference is caused by bucketing)")
        self.logger.print(f"device: {log_utils.yellow(vae.device)} | dtype: {log_utils.yellow(vae.dtype)}")
        self.logger.print('async cache enabled')

        pbar = self.logger.tqdm(total=len(batches), desc=f"caching latents", disable=not self.is_main_process)
        for batch in batches:
            cache_batch_latents(batch, vae, cache_to_disk=cache_to_disk, flip_aug=self.flip_aug, cache_only=self.cache_only, empty_cache=empty_cache, async_cache=async_cache)
            pbar.update(1)
        pbar.close()

        accelerator.wait_for_everyone()

        # check if all latents are cached
        for image_info in image_infos:
            if cache_to_disk and image_info.npz_path is None:
                npz_path = image_info.image_path.with_suffix('.npz')
                assert npz_path.exists(), f"npz file still not found: {npz_path}"
                image_info.npz_path = npz_path

        self.logger.print(log_utils.green(f"caching finished at process {self.process_idx}/{self.num_processes}"), disable=False)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch = self.batches[index]
        sample = dict(
            image_keys=[],
            images=[],
            latents=[],
            captions=[],
            target_size_hw=[],
            original_size_hw=[],
            crop_top_lefts=[],
            flipped=[],
            input_ids_1=[],
            input_ids_2=[],
        )

        for img_info in batch:
            img_info: ImageInfo
            flipped = self.flip_aug and random.random() > 0.5
            if img_info.latents is not None:  # directly load latents from memory
                # logu.debug(f"Find latents: {image_info.key}")
                latents = img_info.latents if not flipped else img_info.latents_flipped
                image = None
            elif img_info.npz_path is not None:  # load latents from disk
                # logu.debug(f"Load latents from disk: {image_info.key}")
                latents, latents_flipped, orig_size, crop_ltrb = load_latents_from_disk(img_info.npz_path, flip_aug=self.flip_aug, dtype=self.latents_dtype, is_main_process=self.is_main_process)
                if orig_size is not None:
                    img_info.original_size = orig_size
                if crop_ltrb is not None:
                    img_info.crop_ltrb = crop_ltrb
                if latents is None or (flipped and latents_flipped is None):
                    raise RuntimeError(f"Invalid latents: {img_info.npz_path}")
                if self.keep_cached_latents_in_memory:
                    img_info.latents = latents
                    img_info.latents_flipped = latents_flipped
                if flipped:
                    latents, latents_flipped = latents_flipped, latents
                    del latents_flipped
                image = None
            elif img_info.image_path is not None:  # load image from disk
                # logu.debug(f"Load image from disk: {image_info.key}")
                image = load_image(img_info.image_path)
                image = process_image(image, target_size=img_info.bucket_size)  # (3, H, W)
                if flipped:
                    image = torch.flip(image, dims=[2])
                latents = None
            else:
                # TODO: Implement non-latent-cache training
                raise NotImplementedError("No latents found for image: {}".format(img_info.image_path))

            # if image is None and img_info.image_path is not None:
            #     image = load_image(img_info.image_path)

            target_size = img_info.bucket_size or ((image.shape[0], image.shape[1]) if image is not None else (latents.shape[2]*8, latents.shape[1]*8))
            # assert target_size[0] >= 3 and target_size[
            #     1] >= 3, f"target size must be at least 3, but got target_size: {target_size} | image_shape: {image.shape if image is not None else None} | latents_shape: {latents.shape if latents is not None else None}"

            orig_size = img_info.original_size or img_info.image_size
            crop_ltrb = img_info.crop_ltrb or (0, 0, 0, 0)  # ! temporary set to 0: no crop at all
            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            if img_info.description is not None and img_info.caption is not None:
                caption = self.process_caption(img_info, flip_aug=flipped) if random.random() > 0.5 else self.process_description(img_info, flip_aug=flipped)
                # print(f"caption of {image_info.key}: {caption}")
            elif img_info.description is not None:
                caption = self.process_description(img_info, flip_aug=flipped)
            elif img_info.caption is not None:
                caption = self.process_caption(img_info, flip_aug=flipped)
            else:
                log_utils.warn(f"no caption or tags found for image: {img_info.key}")
                caption = ''

            input_ids_1 = self.get_input_ids(caption, self.tokenizer1)
            input_ids_2 = self.get_input_ids(caption, self.tokenizer2)

            sample["image_keys"].append(img_info.key)
            sample["images"].append(image)
            sample["latents"].append(latents)
            sample["flipped"].append(flipped)
            sample["target_size_hw"].append((target_size[1], target_size[0]))
            sample["original_size_hw"].append((orig_size[1], orig_size[0]))
            sample["crop_top_lefts"].append((crop_left_top[1], crop_left_top[0]))
            sample["captions"].append(caption)
            sample["input_ids_1"].append(input_ids_1)
            sample["input_ids_2"].append(input_ids_2)

        sample["images"] = torch.stack(sample["images"], dim=0).to(memory_format=torch.contiguous_format).float() if sample["images"][0] is not None else None
        sample["latents"] = torch.stack(sample["latents"], dim=0) if sample["latents"][0] is not None else None
        sample["target_size_hw"] = torch.stack([torch.LongTensor(x) for x in sample["target_size_hw"]])
        sample["original_size_hw"] = torch.stack([torch.LongTensor(x) for x in sample["original_size_hw"]])
        sample["crop_top_lefts"] = torch.stack([torch.LongTensor(x) for x in sample["crop_top_lefts"]])
        sample["input_ids_1"] = torch.stack(sample["input_ids_1"], dim=0)
        sample["input_ids_2"] = torch.stack(sample["input_ids_2"], dim=0)

        # if not self.keep_cached_latents_in_memory:
        #     for image_info in batch:
        #         image_info.latents = None
        #         image_info.latents_flipped = None
        #         del latents

        return sample


def listdir(
    directory,
    exts: Optional[Tuple[str]] = None,
    return_type: Optional[type] = None,
    return_path: Optional[bool] = False,
    return_dir: Optional[bool] = False,
    recur: Optional[bool] = False,
    return_abspath: Optional[bool] = True,
):
    r"""
    List files in a directory.
    :param directory: The directory to list files in.
    :param exts: The extensions to filter by. If None, all files are returned.
    :param return_type: The type to return the files as. If None, returns the type of the directory. If return_path is True, returns str anyway.
    :param return_path: Whether to return the full path of the files.
    :param return_dir: Whether to return directories instead of files.
    :param recur: Whether to recursively list files in subdirectories.
    :param return_abspath: Whether to return absolute paths.
    :return: A list of files in the directory.
    """
    if exts and return_dir:
        raise ValueError("Cannot return both files and directories")

    if not return_path and return_type and return_type != str:
        raise ValueError("Cannot return non-str type when returning name")

    if not recur:
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
    else:
        files = []
        for root, dirs, filenames in os.walk(directory):
            for f in filenames:
                files.append(os.path.join(root, f))

    if exts:
        files = [f for f in files if os.path.splitext(f)[1] in exts]
    if return_dir:
        files = [f for f in files if os.path.isdir(f)]
    if not return_path:
        files = [os.path.basename(f) for f in files]
    if return_abspath:
        files = [os.path.abspath(f) for f in files]
    if return_type == Path:
        files = [return_type(f) for f in files]

    return files


def search_image_file(image_dir, image_key):
    image_key, ext = os.path.splitext(image_key)
    if ext != '':
        image_file = os.path.join(image_dir, f"{image_key}{ext}")
        if os.path.isfile(image_file):
            image_path = os.path.abspath(image_file)
            return image_path
        else:
            image_path = None
    for ext in IMAGE_EXTENSIONS:
        image_file = os.path.join(image_dir, f"{image_key}{ext}")
        if os.path.isfile(image_file):
            image_path = os.path.abspath(image_file)
            break
    else:
        image_path = None
    return image_path


def search_cache_file(cache_dir, image_key):
    cache_file = os.path.join(cache_dir, f"{image_key}.npz")
    if os.path.isfile(cache_file):
        cache_path = os.path.abspath(cache_file)
    else:
        cache_path = None
    return cache_path


def search_file(paths, exts):
    for path in paths:
        if path.suffix in exts:
            return path


def around_reso(img_w, img_h, reso: Union[Tuple[int, int], int], divisible: Optional[int] = None) -> Tuple[int, int]:
    r"""
    w*h = reso*reso
    w/h = img_w/img_h
    => w = img_ar*h
    => img_ar*h^2 = reso
    => h = sqrt(reso / img_ar)
    """
    reso = reso if isinstance(reso, tuple) else (reso, reso)
    divisible = divisible or 1
    img_ar = img_w / img_h
    around_h = int(math.sqrt(reso[0]*reso[1] /
                   img_ar) // divisible * divisible)
    around_w = int(img_ar * around_h // divisible * divisible)
    return (around_w, around_h)


def aspect_ratio_diff(size_1: Tuple[int, int], size_2: Tuple[int, int]):
    ar_1 = size_1[0] / size_1[1]
    ar_2 = size_2[0] / size_2[1]
    return max(ar_1/ar_2, ar_2/ar_1)


def rotate_image_straight(image: Image) -> Image:
    exif: Image.Exif = image.getexif()
    if exif:
        orientation_tag = {v: k for k, v in ExifTags.TAGS.items()}[
            'Orientation']
        orientation = exif.get(orientation_tag)
        degree = {
            3: 180,
            6: 270,
            8: 90,
        }.get(orientation)
        if degree:
            image = image.rotate(degree, expand=True)
    return image


def closest_resolution(buckets: List[Tuple[int, int]], size: Tuple[int, int]) -> Tuple[int, int]:
    img_ar = size[0] / size[1]

    def distance(reso: Tuple[int, int]) -> float:
        return abs(img_ar - reso[0]/reso[1])

    return min(buckets, key=distance)


def get_bucket_reso(
    image_size: Tuple[int, int],
    buckets: Optional[List[Tuple[int, int]]] = SDXL_BUCKET_RESOS,
    max_resolution: Optional[Union[Tuple[int, int], int]] = 1024,
    max_aspect_ratio: Optional[float] = 1.1,
    divisible: Optional[int] = 32
):
    r"""
    Get the closest resolution to the image's resolution from the buckets. If the image's aspect ratio is too
    different from the closest resolution, then return the around resolution based on the max resolution.
    :param image: The image to be resized.
    :param buckets: The buckets of resolutions to choose from. Default to SDXL_BUCKETS. Set None to use max_resolution.
    :param max_resolution: The max resolution to be used to calculate the around resolution. It's used to calculate the 
        around resolution when `buckets` is None or no bucket can contain that image without exceeding the max aspect ratio.
        Default to 1024. Set `-1` to auto calculate from the buckets' max resolution. Set None to disable.
        Set None to auto calculate from the buckets' max resolution.
    :param max_aspect_ratio: The max aspect ratio difference between the image and the closest resolution. Default to 1.1.
        Set None to disable.
    :param divisible: The divisible number of bucket resolutions. Default to 32.
    :return: The closest resolution to the image's resolution.
    """
    if not buckets and (not max_resolution or max_resolution == -1):
        raise ValueError(
            "Either `buckets` or `max_resolution` must be provided.")

    img_w, img_h = image_size
    clo_reso = closest_resolution(buckets, image_size) if buckets else around_reso(
        img_w, img_h, reso=max_resolution, divisible=divisible)
    max_resolution = max(
        buckets, key=lambda x: x[0]*x[1]) if buckets and max_resolution == -1 else max_resolution

    # Handle special resolutions
    if img_w < clo_reso[0] or img_h < clo_reso[1]:
        new_w = img_w // divisible * divisible
        new_h = img_h // divisible * divisible
        clo_reso = (new_w, new_h)
    elif max_aspect_ratio and aspect_ratio_diff((img_w, img_h), clo_reso) >= max_aspect_ratio:
        if buckets and max_resolution:
            clo_reso = around_reso(
                img_w, img_h, reso=max_resolution, divisible=divisible)
        else:
            log_utils.warn(
                f"An image has aspect ratio {img_w/img_h:.2f} which is too different from the closest resolution {clo_reso[0]/clo_reso[1]}. You may lower the `divisible` to avoid this.")

    return clo_reso


def caption2metadata(caption):
    tags = caption.split(',')
    tags = [tag.strip() for tag in tags]
    metadata = {
        'caption': caption,
    }
    for tag in tags:
        if tag.startswith('artist:'):
            metadata['artist'] = tag[7:].strip()
        elif tag.startswith('character:'):
            metadata.setdefault('characters', []).append(tag[10:].strip())
        elif tag.startswith('style:'):
            metadata.setdefault('styles', []).append(tag[6:].strip())
        elif 'quality' in tag:
            metadata['quality'] = tag.replace('quality', '').strip()
    if 'characters' in metadata:
        metadata['characters'] = ', '.join(metadata['characters'])
    if 'styles' in metadata:
        metadata['styles'] = ', '.join(metadata['styles'])
    return metadata


def empty_metadata():
    return {
        'image_path': None,
        'caption': None,
        'description': None,
        'artist': None,
        'characters': None,
        'styles': None,
        'quality': None,
        'original_size': None,
        'safe_rating': None,
        'safe_level': None,
        'perceptual_hash': None,
        'aesthetic_score': None,
    }


def get_image_size(image_path):
    image = Image.open(image_path)
    return image.size


def get_latent_image_size(npz_path):
    npz = open_cache(npz_path, mmap_mode='r')
    if npz is None:
        return None
    latents = npz["latents"]
    latent_image_size = latents.shape[-1], latents.shape[-2]
    return latent_image_size


def load_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    img = np.array(image, np.uint8)  # (H, W, C)
    return img


def make_canny(image: np.ndarray, thres_1=0, thres_2=75):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    canny_edges = cv2.Canny(blurred_img, thres_1, thres_2)
    return canny_edges


def resize_if_needed(image, target_size):
    if image.shape[0] != target_size[1] or image.shape[1] != target_size[0]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image


def process_image(image, target_size):
    image = resize_if_needed(image, target_size)
    image = IMAGE_TRANSFORMS(image)
    return image


def open_cache(npz_path, mmap_mode=None, is_main_process=True):
    try:
        npz = np.load(npz_path, mmap_mode=mmap_mode)
        return npz
    except Exception as e:
        if is_main_process:
            import shutil
            backup_path = str(npz_path) + '.bak'
            shutil.move(str(npz_path), backup_path)
            print(f"remove corrupted npz file: {os.path.abspath(npz_path)}")
            print(f"  error: {e}")
        return None


def load_latents_from_disk(npz_path, dtype=None, flip_aug=True, mmap_mode=None, is_main_process=True):
    npz = open_cache(npz_path, mmap_mode=mmap_mode, is_main_process=is_main_process)
    if npz is None:
        return None, None, None, None
    latents = npz["latents"]
    flipped_latents = npz["latents_flipped"] if "latents_flipped" in npz else None
    orig_size = npz["original_size"].tolist() if "original_size" in npz else None
    crop_ltrb = npz["crop_ltrb"].tolist() if "crop_ltrb" in npz else None
    latents = torch.FloatTensor(latents).to(dtype=dtype)
    flipped_latents = torch.FloatTensor(flipped_latents).to(dtype=dtype) if flipped_latents is not None else None

    if torch.any(torch.isnan(latents)):
        latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        print(f"NaN detected in latents: {npz_path}")
    if flipped_latents is not None and torch.any(torch.isnan(flipped_latents)):
        flipped_latents = torch.where(torch.isnan(flipped_latents), torch.zeros_like(flipped_latents), flipped_latents)
        print(f"NaN detected in flipped latents: {npz_path}")

    return latents, flipped_latents, orig_size, crop_ltrb


def check_cached_latents(image_info, latents):
    if latents is None:
        return False
    bucket_reso_hw = (image_info.bucket_size[1], image_info.bucket_size[0])
    latents_hw = (latents.shape[1]*8, latents.shape[2]*8)
    # assert latents_hw[0] > 128 and latents_hw[1] > 128, f"latents_hw must be larger than 128, but got latents_hw: {latents_hw}"
    # print(f"  bucket_reso_hw: {bucket_reso_hw} | latents_hw: {latents_hw}")
    return latents_hw == bucket_reso_hw


def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None):
    kwargs = {}
    if flipped_latents_tensor is not None:
        kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
    try:
        np.savez(
            npz_path,
            latents=latents_tensor.float().cpu().numpy(),
            original_size=np.array(original_size),
            crop_ltrb=np.array(crop_ltrb),
            **kwargs,
        )
    except KeyboardInterrupt:
        raise
    if not os.path.isfile(npz_path):
        raise RuntimeError(f"Failed to save latents to {npz_path}")


async def save_latents_to_disk_async(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None):
    from io import BytesIO
    kwargs = {}
    if flipped_latents_tensor is not None:
        kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()

    # 使用BytesIO作为临时内存文件
    buffer = BytesIO()
    np.savez(
        buffer,
        latents=latents_tensor.float().cpu().numpy(),
        original_size=np.array(original_size),
        crop_ltrb=np.array(crop_ltrb),
        **kwargs,
    )

    # 将BytesIO缓冲区的内容异步写入磁盘
    buffer.seek(0)  # 重置buffer的位置到开始
    async with aiofiles.open(npz_path, 'wb') as f:
        await f.write(buffer.read())

    # 检查文件是否真的被写入
    if not os.path.isfile(npz_path):
        raise RuntimeError(f"Failed to save latents to {npz_path}")


def cache_batch_latents(image_infos: List[ImageInfo], vae, cache_to_disk, flip_aug, cache_only=False, async_cache=False, empty_cache=False):
    images = []
    for info in image_infos:
        image = load_image(info.image_path)
        image = process_image(image, target_size=info.bucket_size)
        images.append(image)

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().to('cpu')

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent in zip(image_infos, latents, flipped_latents):
        # check NaN
        if torch.isnan(latents).any() or (flipped_latent is not None and torch.isnan(flipped_latent).any()):
            raise RuntimeError(f"NaN detected in latents: {info.absolute_path}")

    if not async_cache:
        for info, latent, flipped_latent in zip(image_infos, latents, flipped_latents):
            # check NaN
            if torch.isnan(latents).any() or (flipped_latent is not None and torch.isnan(flipped_latent).any()):
                raise RuntimeError(f"NaN detected in latents: {info.absolute_path}")

            if cache_to_disk:
                npz_path = os.path.splitext(info.image_path)[0] + ".npz"
                orig_size = info.original_size or info.image_size
                crop_ltrb = (0, 0, 0, 0)  # ! temporary set to 0: no crop at all
                save_latents_to_disk(npz_path, latent, original_size=orig_size, crop_ltrb=crop_ltrb, flipped_latents_tensor=flipped_latent)
                info.npz_path = npz_path

            if not cache_only:
                info.latents = latent
                if flip_aug:
                    info.latents_flipped = flipped_latent
    else:
        async def save_latents():
            tasks = []
            iterator = zip(image_infos, latents, flipped_latents)
            if cache_to_disk:
                for info, latent, flipped_latent in iterator:
                    npz_path = os.path.splitext(info.image_path)[0] + ".npz"
                    orig_size = info.original_size or info.image_size
                    crop_ltrb = (0, 0, 0, 0)  # ! temporary set to 0: no crop at all
                    task = asyncio.create_task(save_latents_to_disk_async(npz_path, latent, original_size=orig_size, crop_ltrb=crop_ltrb, flipped_latents_tensor=flipped_latent))
                    info.npz_path = npz_path
                    tasks.append(task)

            if not cache_only:
                for info, latent, flipped_latent in iterator:
                    info.latents = latent
                    if flip_aug:
                        info.latents_flipped = flipped_latent

            await asyncio.gather(*tasks)
        asyncio.run(save_latents())

    # FIXME this slows down caching a lot, specify this as an option
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_input_ids(caption, tokenizer, max_token_length):
    input_ids = tokenizer(
        caption, padding="max_length", truncation=True, max_length=max_token_length, return_tensors="pt"
    ).input_ids

    if max_token_length > tokenizer.model_max_length:
        input_ids = input_ids.squeeze(0)
        iids_list = []
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            # v1
            # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
            # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
            for i in range(
                1, max_token_length - tokenizer.model_max_length +
                    2, tokenizer.model_max_length - 2
            ):  # (1, 152, 75)
                ids_chunk = (
                    input_ids[0].unsqueeze(0),
                    input_ids[i: i + tokenizer.model_max_length - 2],
                    input_ids[-1].unsqueeze(0),
                )
                ids_chunk = torch.cat(ids_chunk)
                iids_list.append(ids_chunk)
        else:
            # v2 or SDXL
            # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
            for i in range(1, max_token_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                ids_chunk = (
                    input_ids[0].unsqueeze(0),  # BOS
                    input_ids[i: i + tokenizer.model_max_length - 2],
                    input_ids[-1].unsqueeze(0),
                )  # PAD or EOS
                ids_chunk = torch.cat(ids_chunk)

                # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                    ids_chunk[-1] = tokenizer.eos_token_id
                # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                if ids_chunk[1] == tokenizer.pad_token_id:
                    ids_chunk[1] = tokenizer.eos_token_id

                iids_list.append(ids_chunk)

        input_ids = torch.stack(iids_list)  # 3,77
        return input_ids


def fmt2dan(tag):
    if isinstance(tag, str):
        tag = tag.lower().strip()
        tag = tag.replace(': ', ':').replace(' ', '_').replace('\\(', '(').replace('\\)', ')')
        return tag
    elif isinstance(tag, list):
        return [fmt2dan(t) for t in tag]
    else:
        return tag


def count_metadata(metadata):
    r"""
    Count useful information from metadata.
    """
    counter = {
        "category": {},
        "artist": {},
        "character": {},
        "style": {},
        "quality": {},
    }
    for img_key, img_md in metadata.items():
        num_repeats = img_md.get('num_repeats', 1)
        src_path = img_md.get('image_path') or img_md.get('npz_path')
        category = os.path.basename(os.path.dirname(src_path))
        if category not in counter["category"]:
            counter["category"][category] = 0
        counter["category"][category] += num_repeats

        artist, characters, styles, quality = img_md.get('artist'), img_md.get('characters'), img_md.get('styles'), img_md.get('quality')

        if artist:
            artist = fmt2dan(artist)
            if artist not in counter["artist"]:
                counter["artist"][artist] = 0
            counter["artist"][artist] += num_repeats
        if characters:
            for character in characters.split(', '):
                character = fmt2dan(character)
                if character not in counter["character"]:
                    counter["character"][character] = 0
                counter["character"][character] += num_repeats
        if styles:
            for style in styles.split(', '):
                style = fmt2dan(style)
                if style not in counter["style"]:
                    counter["style"][style] = 0
                counter["style"][style] += num_repeats
        if quality:
            if quality not in counter["quality"]:
                counter["quality"][quality] = 0
            counter["quality"][quality] += num_repeats
    # sort
    for key in counter:
        counter[key] = dict(sorted(counter[key].items(), key=lambda x: x[1], reverse=True))

    return counter
