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
from tqdm import tqdm
from PIL import Image, ExifTags
from typing import List, Tuple, Optional, Union
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, wait
from . import custom_train_utils, log_utils as logu

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
        image_key: str,
        abs_path,
        caption: str = None,
        nl_caption: str = None,
        image_size=None,  # (w, h)
        original_size=None,  # (w, h)
        latents_size=None,  # (w, h)
        bucket_reso=None,
        num_repeats=1,
        npz_path=None,
        latents=None,
        latents_flipped=None,
        metadata=None,
        control_images=None,
    ):
        self.image_key = image_key
        self.caption = caption
        self.nl_caption = nl_caption
        self.abs_path = abs_path
        self.image_size = image_size
        self.orig_size = original_size
        self.latents_size = latents_size
        self.bucket_reso = bucket_reso
        self.num_repeats = num_repeats
        self.npz_path = npz_path
        self.latents = latents
        self.latents_flipped = latents_flipped
        self.metadata = metadata
        self.control_images = control_images


def search_image_file(image_dir, image_key):
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
            logu.warn(
                f"An image has aspect ratio {img_w/img_h:.2f} which is too different from the closest resolution {clo_reso[0]/clo_reso[1]}. You may lower the `divisible` to avoid this.")

    return clo_reso


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        tokenizer1,
        tokenizer2,
        latents_dtype=torch.float32,
        predefined_bucket_resos=SDXL_BUCKET_RESOS,
        is_main_process=False,
        num_processes=1,
        process_idx=0,
    ):
        self.image_dir = Path(args.image_dir).absolute()
        self.metadata_file = Path(args.metadata_file).absolute()
        self.recording_dir = Path(args.recording_dir).absolute() if args.recording_dir else None
        self.batch_size = args.batch_size
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2

        self.latents_dtype = latents_dtype
        self.max_token_length = args.max_token_length
        self.predefined_bucket_resos = predefined_bucket_resos

        self.flip_aug = args.flip_aug
        self.tags_shuffle_prob = args.tags_shuffle_prob
        self.tags_shuffle_rate = args.tags_shuffle_rate
        self.fixed_tag_dropout_rate = args.fixed_tag_dropout_rate
        self.flex_tag_dropout_rate = args.flex_tag_dropout_rate

        self.resolution = args.resolution
        self.bucket_reso_step = args.bucket_reso_step

        self.check_cache_validity = args.check_cache_validity
        self.keep_cached_latents_in_memory = args.keep_cached_latents_in_memory

        self.max_workers = min(args.max_dataset_n_workers, os.cpu_count() - 1)
        self.is_main_process = is_main_process
        self.num_processes = num_processes
        self.process_idx = process_idx

        self.image_data = {}
        self.buckets = {}
        self.batches = []

        if self.metadata_file.is_file():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(
                "Metadata file not found: {}".format(self.metadata_file))

        self.num_train_images = 0
        self.num_train_repeats = 0

        # load img_size_record
        if self.recording_dir:
            img_sz_log_path = self.recording_dir / "image_size.json"
            img_size_record = {}
            if img_sz_log_path.is_file():
                if is_main_process:
                    print(logu.green(f"use existing `image size` record: {img_sz_log_path}"))
                with open(img_sz_log_path) as f:
                    img_size_record = json.load(f)
            else:
                if is_main_process:
                    print(logu.yellow(f"no existing `image size` record found: {img_sz_log_path}"))

            # load num_repeats_record
            n_rep_log_path = self.recording_dir / "num_repeats.json"
            num_repeats_record = {}
            # if n_rep_log_path.is_file():
            #     if is_main_process:
            #         print(logu.green(f"use existing `num repeats` record: {n_rep_log_path}"))
            #     with open(n_rep_log_path) as f:
            #         num_repeats_record = json.load(f)
            # else:
            #     if is_main_process:
            #         print(logu.yellow(f"no existing `num repeats` record found: {n_rep_log_path}"))

        if (reg_md_path := self.metadata_file.with_name('reg_metadata.json')).is_file():
            with open(reg_md_path, 'r') as f:
                reg_metadata = json.load(f)
            if is_main_process:
                print(logu.green(f"use regularizing metadata from: {reg_md_path} | size: {logu.yellow(len(reg_metadata))}"))
        else:
            reg_metadata = {}
            if is_main_process:
                print(logu.yellow(f"no existing reg metadata found: {reg_md_path}"))

        counter = custom_train_utils.count_metadata(self.metadata)
        self.counter = counter
        artist_benchmark = 100
        # artist_benchmark = sum(counter["artist"].values()) / len(counter["artist"]) if len(counter["artist"]) > 0 else None  # average number of images per artist

        if is_main_process:
            print(f"artist benchmark for calculating num_repeats: {artist_benchmark:.2f}")

        pbar = tqdm(total=len(self.metadata), desc='loading dataset', disable=not is_main_process)
        logs = {}

        @logu.track_tqdm(pbar)
        def load_data(img_key, img_md):
            # 1. search image or cache file
            image_path = search_image_file(self.image_dir, img_key)
            npz_path = search_cache_file(self.image_dir, img_key)
            if image_path is None and npz_path is None:
                return

            # 2. get caption
            caption = img_md.get('caption', None)  # tag caption
            nl_caption = img_md.get('nl_caption', None)  # natural language caption

            # 3. get num_repeats
            # num_repeats = img_md.get('num_repeats', None) or num_repeats_record.get(img_key, None)
            # if num_repeats is None:
            num_repeats = custom_train_utils.get_num_repeats(img_key, img_md, counter, artist_benchmark, reg_metadata)
            if num_repeats == 0:  # drop
                return

            # 4. get image size
            # image_size = img_md.get("image_size", None)
            image_size = None
            latents_size = None
            bucket_reso = None
            if image_size:  # if image_size is provided, pass
                pass
            elif img_key in img_size_record:  # if image_size is cached, use it
                image_size = tuple(img_size_record[img_key])
            elif image_path is not None and (not npz_path or (npz_path and self.check_cache_validity)):  # if image_size is not provided, try to read from image file
                image_size = get_image_size(image_path)
            else:
                latents_size = get_latent_image_size(npz_path)
                if latents_size is None:
                    raise RuntimeError(f"failed to read image size: `{img_key}`. Please check if the image file exists or the cached latent is valid.")
                image_size = (latents_size[0] * 8, latents_size[1] * 8)
                bucket_reso = image_size  # ! directly use latents_size as bucket_reso

            original_size = img_md.get('original_size', None)

            # 5. record and log
            if self.recording_dir:
                num_repeats_record[img_key] = num_repeats
                img_size_record[img_key] = image_size

            artist = img_md.get('artist', None)
            if artist is not None and artist not in logs:
                logs[artist] = num_repeats

            # if npz_path:
            #     latents, latents_flipped = load_latents_from_disk(npz_path, dtype=self.dtype)
            #     if latents is None:
            #         raise RuntimeError(f"failed to load latents from disk: `{img_key}`. Please check if the cached latent is valid.")
            #     if latents_flipped is None and self.flip_aug:
            #         raise RuntimeError(f"failed to load flipped latents from disk: `{img_key}`. Please check if the cached latent is valid.")

            img_info = ImageInfo(
                image_key=img_key,
                caption=caption,
                nl_caption=nl_caption,
                abs_path=image_path,
                npz_path=npz_path,
                image_size=image_size,
                original_size=original_size,
                latents_size=latents_size,
                bucket_reso=bucket_reso,
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
            print(f"num_train_images: {logu.yellow(self.num_train_images)} | num_train_repeats: {logu.yellow(self.num_train_repeats)}")
            print(f"quality counter:", ' | '.join([f"{k}: {logu.yellow(v)}" for k, v in counter['quality'].items()]))
            if self.recording_dir:
                n_rep_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(n_rep_log_path, 'w') as f:
                    json.dump(num_repeats_record, f)
                img_sz_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(img_sz_log_path, 'w') as f:
                    json.dump(img_size_record, f)
                print(f"records saved to `{logu.yellow(self.recording_dir)}`")

        self.make_buckets()
        self.make_batches()

    def register_image_info(self, image_info):
        self.image_data[image_info.image_key] = image_info

    def process_caption(self, image_info: ImageInfo):
        return custom_train_utils.process_caption(
            image_info.caption,
            image_info.metadata,
            self.counter,
            fixed_tag_dropout_rate=self.fixed_tag_dropout_rate,
            flex_tag_dropout_rate=self.flex_tag_dropout_rate,
            tags_shuffle_prob=self.tags_shuffle_prob,
            tags_shuffle_rate=self.tags_shuffle_rate,
        )

    def process_nl_caption(self, image_info: ImageInfo, flip_aug=False):
        return custom_train_utils.process_nl_caption(
            image_info.nl_caption,
            flip_aug=flip_aug,
        )

    def get_input_ids(self, caption, tokenizer):
        return get_input_ids(caption, tokenizer, max_token_length=self.max_token_length)

    def shuffle_buckets(self):
        # random.shuffle(self.buckets)
        for bucket in self.buckets.values():
            random.shuffle(bucket)

    def make_buckets(self):
        for img_info in tqdm(self.image_data.values(), desc='making buckets', disable=not self.is_main_process):
            if img_info.bucket_reso is not None:
                pass
            elif img_info.latents_size is not None or img_info.latents is not None:
                if img_info.latents is not None:  # latents.shape: [C, H, W]
                    bucket_reso = (img_info.latents.shape[-1] * 8, img_info.latents.shape[-2] * 8)
                else:
                    bucket_reso = (img_info.latents_size[0] * 8, img_info.latents_size[1] * 8)
                img_info.bucket_reso = bucket_reso
                # assert expected_bucket_reso == img_info.bucket_reso, f"latent size and bucket reso of `{img_info.image_key}` mismatch: excepted bucket reso to be {expected_bucket_reso}, but got {img_info.bucket_reso}"
            else:  # make from image file
                bucket_reso = get_bucket_reso(img_info.image_size, buckets=self.predefined_bucket_resos,
                                              max_resolution=self.resolution, divisible=self.bucket_reso_step)
                img_info.bucket_reso = bucket_reso

            assert img_info.bucket_reso[0] % self.bucket_reso_step == 0 and img_info.bucket_reso[1] % self.bucket_reso_step == 0, \
                f"bucket reso must be divisible by {self.bucket_reso_step}: {img_info.bucket_reso}"

            if img_info.bucket_reso not in self.buckets:
                self.buckets[img_info.bucket_reso] = []
            self.buckets[img_info.bucket_reso].extend([img_info] * img_info.num_repeats)

        self.shuffle_buckets()

    def debug_buckets(self):
        for i, bucket_reso in enumerate(self.buckets):
            bucket = self.buckets[bucket_reso]
            for j, img_info in enumerate(bucket):
                print(f"  [{j}]: {img_info.image_key} | {img_info.image_size} -> {img_info.bucket_reso}")

    def make_batches(self):
        self.batches = []
        for bucket in tqdm(self.buckets.values(), desc='making batches', disable=not self.is_main_process):
            for i in range(0, len(bucket), self.batch_size):
                self.batches.append(bucket[i:i+self.batch_size])

    # def make_depth(self):
    #     ...

    # def make_canny(self):
    #     for img_info in tqdm(self.image_data.values(), desc='making canny', disable=not self.is_main_process):
    #         if img_info.abs_path is not None:
    #             image = load_image(img_info.abs_path)
    #             image = process_image(image, target_size=img_info.bucket_reso)
    #             canny = make_canny(image)
    #             img_info.canny = canny

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, check_validity=False, cache_only=False, empty_cache=False, async_cache=False):
        image_infos = list(self.image_data.values())
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])
        batches = []
        batch = []
        pbar = tqdm(total=len(image_infos), desc='checking latents', disable=not self.is_main_process)
        pbar_logs = {
            'invalid': 0,
            'miss': 0,
        }
        pbar.set_postfix(pbar_logs)
        for image_info in image_infos:
            if not image_info.abs_path or (image_info.latents is not None and (not self.flip_aug or image_info.latents_flipped is not None)):
                pbar.update(1)
                continue

            if image_info.npz_path:  # if npz file exists
                if not check_validity:
                    pbar.update(1)
                    continue
                latents, latents_flipped = load_latents_from_disk(image_info.npz_path, flip_aug=self.flip_aug, dtype=self.latents_dtype)
                image_info.latents = latents
                image_info.latents_flipped = latents_flipped
                # print(f"check latents: {image_info.image_key} | {image_info.npz_path}")
                if check_cached_latents(image_info, latents) and (not self.flip_aug or check_cached_latents(image_info, latents_flipped)):  # if latents is valid
                    pbar.update(1)
                    continue
                else:
                    pbar_logs['miss'] += 1
                    pbar.set_postfix(pbar_logs)
            else:
                pbar_logs['invalid'] += 1
                pbar.set_postfix(pbar_logs)

            if len(batch) > 0 and batch[-1].bucket_reso != image_info.bucket_reso:
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

        if self.num_processes > 1:
            batches = batches[self.process_idx::self.num_processes]  # split batches into processes

        pbar = tqdm(total=len(batches), desc='caching latents', disable=not self.is_main_process)
        for batch in batches:
            cache_batch_latents(batch, vae, cache_to_disk=cache_to_disk, flip_aug=self.flip_aug, cache_only=cache_only, empty_cache=empty_cache, async_cache=async_cache)
            pbar.update(1)
        pbar.close()

        logu.success(f"caching finished at process {self.process_idx}/{self.num_processes}")

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
            control_images=[],
        )

        for image_info in batch:
            image_info: ImageInfo
            flipped = self.flip_aug and random.random() > 0.5
            if image_info.latents is not None:  # directly load latents from memory
                # logu.debug(f"Find latents: {image_info.image_key}")
                latents = image_info.latents if not flipped else image_info.latents_flipped
                image = None
            elif image_info.npz_path is not None:  # load latents from disk
                # logu.debug(f"Load latents from disk: {image_info.image_key}")
                latents, latents_flipped = load_latents_from_disk(image_info.npz_path, flip_aug=self.flip_aug, dtype=self.latents_dtype)
                if latents is None or (flipped and latents_flipped is None):
                    raise RuntimeError(f"Invalid latents: {image_info.npz_path}")
                if self.keep_cached_latents_in_memory:
                    image_info.latents = latents
                    image_info.latents_flipped = latents_flipped
                if flipped:
                    latents, latents_flipped = latents_flipped, latents
                    del latents_flipped
                image = None
            elif image_info.abs_path is not None:  # load image from disk
                # logu.debug(f"Load image from disk: {image_info.image_key}")
                image = load_image(image_info.abs_path)
                image = process_image(image, target_size=image_info.bucket_reso)  # (3, H, W)
                if flipped:
                    image = torch.flip(image, dims=[2])
                latents = None
            else:
                # TODO: Implement non-latent-cache training
                raise NotImplementedError("No latents found for image: {}".format(image_info.abs_path))

            if image is None and image_info.abs_path is not None:
                image = load_image(image_info.abs_path)

            target_size = image_info.bucket_reso or ((image.shape[0], image.shape[1]) if image is not None else (latents.shape[2]*8, latents.shape[1]*8))
            assert target_size[0] > 128 and target_size[
                1] > 128, f"target size must be larger than 128, but got target_size: {target_size} | image_shape: {image.shape if image is not None else None} | latents_shape: {latents.shape if latents is not None else None}"

            orig_size = image_info.orig_size or image_info.image_size
            crop_ltrb = (0, 0, 0, 0)  # ! temporary set to 0: no crop at all
            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            if image_info.nl_caption is not None and image_info.caption is not None:
                caption = self.process_caption(image_info) if random.random() > 0.5 else self.process_nl_caption(image_info, flip_aug=flipped)
                # print(f"caption of {image_info.image_key}: {caption}")
            elif image_info.nl_caption is not None:
                caption = self.process_nl_caption(image_info, flip_aug=flipped)
            elif image_info.caption is not None:
                caption = self.process_caption(image_info)
            else:
                logu.warn(f"no caption or tags found for image: {image_info.image_key}")
                caption = ''

            input_ids_1 = self.get_input_ids(caption, self.tokenizer1)
            input_ids_2 = self.get_input_ids(caption, self.tokenizer2)

            sample["image_keys"].append(image_info.image_key)
            sample["images"].append(image)
            sample["latents"].append(latents)
            sample["flipped"].append(flipped)
            sample["target_size_hw"].append((target_size[1], target_size[0]))
            sample["original_size_hw"].append((orig_size[1], orig_size[0]))
            sample["crop_top_lefts"].append((crop_left_top[1], crop_left_top[0]))
            sample["captions"].append(caption)
            sample["input_ids_1"].append(input_ids_1)
            sample["input_ids_2"].append(input_ids_2)

        sample["latents"] = torch.stack(sample["latents"], dim=0)
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


def open_cache(npz_path, mmap_mode=None):
    try:
        npz = np.load(npz_path, mmap_mode=mmap_mode)
        return npz
    except Exception as e:
        import shutil
        backup_path = npz_path + '.bak'
        shutil.move(npz_path, backup_path)
        print(f"remove corrupted npz file: {os.path.abspath(npz_path)}")
        return None


def load_latents_from_disk(npz_path, dtype=None, flip_aug=True, mmap_mode=None):
    npz = open_cache(npz_path, mmap_mode=mmap_mode)
    if npz is None:
        return None, None
    latents = npz["latents"]
    flipped_latents = npz["latents_flipped"] if "latents_flipped" in npz else None
    latents = torch.FloatTensor(latents).to(dtype=dtype)
    flipped_latents = torch.FloatTensor(flipped_latents).to(dtype=dtype) if flipped_latents is not None else None

    if torch.any(torch.isnan(latents)):
        latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        print(f"NaN detected in latents: {npz_path}")
    if flipped_latents is not None and torch.any(torch.isnan(flipped_latents)):
        flipped_latents = torch.where(torch.isnan(flipped_latents), torch.zeros_like(flipped_latents), flipped_latents)
        print(f"NaN detected in flipped latents: {npz_path}")

    return latents, flipped_latents


def check_cached_latents(image_info, latents):
    if latents is None:
        return False
    bucket_reso_hw = (image_info.bucket_reso[1], image_info.bucket_reso[0])
    latents_hw = (latents.shape[1]*8, latents.shape[2]*8)
    assert latents_hw[0] > 128 and latents_hw[1] > 128, f"latents_hw must be larger than 128, but got latents_hw: {latents_hw}"
    # print(f"  bucket_reso_hw: {bucket_reso_hw} | latents_hw: {latents_hw}")
    return latents_hw == bucket_reso_hw


def save_latents_to_disk(npz_path, latents_tensor, flipped_latents_tensor=None):
    kwargs = {}
    if flipped_latents_tensor is not None:
        kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
    try:
        np.savez(
            npz_path,
            latents=latents_tensor.float().cpu().numpy(),
            **kwargs,
        )
    except KeyboardInterrupt:
        raise
    if not os.path.isfile(npz_path):
        raise RuntimeError(f"Failed to save latents to {npz_path}")


async def save_latents_to_disk_async(npz_path, latents_tensor, flipped_latents_tensor=None):
    from io import BytesIO
    kwargs = {}
    if flipped_latents_tensor is not None:
        kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()

    # 使用BytesIO作为临时内存文件
    buffer = BytesIO()
    np.savez(buffer, latents=latents_tensor.float().cpu().numpy(), **kwargs)

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
        image = load_image(info.abs_path)
        image = process_image(image, target_size=info.bucket_reso)
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
                npz_path = os.path.splitext(info.abs_path)[0] + ".npz"
                save_latents_to_disk(npz_path, latent, flipped_latent)
                info.npz_path = npz_path

            if not cache_only:
                info.latents = latent
                if flip_aug:
                    info.latents_flipped = flipped_latent
    else:
        async def save_latents():
            tasks = []
            iterator = zip(image_infos, latents, flipped_latents)
            for info, latent, flipped_latent in iterator:
                if cache_to_disk:
                    npz_path = os.path.splitext(info.abs_path)[0] + ".npz"
                    task = asyncio.create_task(save_latents_to_disk_async(npz_path, latent, flipped_latent))
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
