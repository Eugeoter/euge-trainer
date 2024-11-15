import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Any
from waifuset import Data, HuggingFaceData
from waifuset import logging

logger = logging.get_logger('dataset')


class CacheLatentsMixin(object):
    cache_latents: bool = False
    vae_batch_size: int = 4
    cache_to_disk: bool = True
    cache_only: bool = False
    async_cache: bool = False
    check_cache_validity: bool = True
    cache_latents_max_dataloader_n_workers: int = 4
    keep_cached_latents_in_memory: bool = True

    def _print_cache_latents_message(self):
        self.logger.print(logging.green(f"==================== START CACHING ===================="))
        self.logger.print(f"cache_latents: {self.cache_latents} | cache_to_disk: {self.cache_to_disk} | cache_only: {self.cache_only}")
        self.logger.print(f"async_cache: {self.async_cache} | check_cache_validity: {self.check_cache_validity}")
        self.logger.print(f"cache_latents_max_dataloader_n_workers: {self.cache_latents_max_dataloader_n_workers} | keep_cached_latents_in_memory: {self.keep_cached_latents_in_memory}")
        self.logger.print(f"vae_batch_size: {self.vae_batch_size}")
        self.logger.print(f"vae_dtype: {self.vae.dtype}")

    def cache_batch_latents(self, vae, empty_cache=False):
        batches = self.make_cache_batches()
        if not batches:
            return
        if self.accelerator.num_processes > 1:
            batches = batches[self.accelerator.process_index::self.accelerator.num_processes]  # split batches into processes
            self.logger.print(f"process {self.accelerator.process_index+1}/{self.accelerator.num_processes} | num_uncached_batches: {len(batches)}", disable=False)

        class CacheLatentsDataset(Dataset):
            def __init__(self, batches, img_getter):
                self.batches = batches
                self.img_getter = img_getter

            def __len__(self):
                return len(self.batches)

            def __getitem__(self, idx):
                batch = self.batches[idx]
                sample = dict(
                    images=[],
                    img_mds=batch,
                )
                for img_md in batch:
                    image = self.img_getter(img_md)
                    sample['images'].append(image)
                sample['images'] = torch.stack(sample['images'], dim=0)
                return sample

        if batches:
            dataset = CacheLatentsDataset(batches, img_getter=self.get_bucket_image)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=self.cache_latents_max_dataloader_n_workers,
                pin_memory=True,
                collate_fn=self.collate_fn,
                shuffle=True,
            )

            for sample in self.logger.tqdm(dataloader, total=len(dataloader), desc="caching latents"):
                self.cache_sample_latents(sample, vae, empty_cache=empty_cache)
        self.accelerator.wait_for_everyone()

    def cache_sample_latents(self, sample, vae, empty_cache=False):
        batch_img_mds = sample['img_mds']
        batch_img_tensors = sample['images']
        batch_img_tensors = batch_img_tensors.to(device=vae.device, dtype=vae.dtype)

        with torch.no_grad():
            batch_latents = vae.encode(batch_img_tensors).latent_dist.sample().to('cpu')

        if not self.async_cache:
            for img_md, latents in zip(batch_img_mds, batch_latents):
                if self.cache_to_disk:
                    cache_path = self.get_cache_path(img_md)
                    original_size = img_md['original_size'] or img_md['image_size']
                    crop_ltrb = (0, 0, 0, 0)  # ! temporary set to 0: no crop at all
                    save_latents_to_disk(latents, cache_path, original_size=original_size, crop_ltrb=crop_ltrb)
                    img_md['cache_path'] = cache_path

                if not self.cache_only:
                    img_md['latents'] = latents
        else:
            import asyncio

            async def save_latents():
                tasks = []
                if self.cache_to_disk:
                    for img_md, latents in zip(batch_img_mds, batch_latents):
                        cache_path = self.get_cache_path(img_md)
                        original_size = img_md['original_size'] or img_md['image_size']
                        crop_ltrb = (0, 0, 0, 0)  # ! temporary set to 0: no crop at all
                        task = asyncio.create_task(save_latents_to_disk_async(latents, cache_path, original_size=original_size, crop_ltrb=crop_ltrb))
                        img_md['cache_path'] = cache_path
                        tasks.append(task)

                if not self.cache_only:
                    for img_md, latents in zip(batch_img_mds, batch_latents):
                        img_md['latents'] = latents

                await asyncio.gather(*tasks)
            asyncio.run(save_latents())

        # FIXME this slows down caching a lot, specify this as an option
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_cache_supported(self, img_md):
        return not isinstance(img_md, HuggingFaceData)

    def get_cache_path(self, img_md):
        return img_md.get('cache_path') or os.path.splitext(img_md['image_path'])[0] + ".npz"

    def make_cache_batches(self) -> List[Data]:
        img_mds = list(self.dataset.values())
        img_mds.sort(key=lambda img_md: img_md['bucket_size'])
        batches = []
        batch = []
        for img_md in img_mds:
            if self.check_cache_validity:
                if not self.is_cache_supported(img_md):
                    continue
                cache = self.get_cache(img_md, update=self.keep_cached_latents_in_memory)
                latents = cache.get('latents')
                if latents is not None and check_cached_latents(latents, img_md) is True:
                    continue
            if os.path.exists(img_md.get('cache_path', '')):
                continue
            if len(batch) > 0 and batch[-1]['bucket_size'] != img_md['bucket_size']:
                batches.append(batch)
                batch = []
            batch.append(img_md)
            if len(batch) >= self.vae_batch_size:
                batches.append(batch)
                batch = []
        if len(batch) > 0:
            batches.append(batch)
        return batches

    def get_cache(self, img_md, update=False) -> Dict[str, Any]:
        if not self.cache_latents:
            return {}
        elif (latents := img_md.get('latents')) is not None:
            cache = {'latents': latents}
        elif (cache_path := self.get_cache_path(img_md)) and os.path.exists(cache_path):
            cache = load_latents_from_disk(cache_path, dtype=self.latents_dtype, is_main_process=self.accelerator.is_main_process)
        else:
            cache = {}
        if update:
            img_md.update(cache)
        return cache

    def get_latents_size(self, img_md):
        if img_md.get('latents') is not None:
            return img_md['latents'].shape[-1], img_md['latents'].shape[-2]
        elif (cache_path := img_md.get('cache_path')) and os.path.exists(cache_path):
            return get_latents_size(cache_path)
        return None


def open_cache(cache_path, mmap_mode=None, is_main_process=True):
    try:
        cache = np.load(cache_path, mmap_mode=mmap_mode)
        return cache
    except Exception as e:
        if is_main_process:
            import shutil
            backup_path = str(cache_path) + '.bak'
            shutil.move(str(cache_path), backup_path)
            logger.print(logging.red(f"remove corrupted cache file: {os.path.abspath(cache_path)}"))
            logger.print(logging.red(f"  error: {e}"))
        return None


def get_latents_size(npz_path):
    npz = open_cache(npz_path, mmap_mode='r')
    if npz is None:
        return None
    latents = npz["latents"]
    latents_size = latents.shape[-1], latents.shape[-2]
    return latents_size


def load_latents_from_disk(cache_path, dtype=None, mmap_mode=None, is_main_process=True) -> Dict[str, Any]:
    cache = open_cache(cache_path, mmap_mode=mmap_mode, is_main_process=is_main_process)
    if cache is None:
        return {}
    latents = cache["latents"]
    extra_info = {}
    if "original_size" in cache:
        extra_info["original_size"] = cache["original_size"].tolist()
    if "crop_ltrb" in cache:
        extra_info["crop_ltrb"] = cache["crop_ltrb"].tolist()
    latents = torch.FloatTensor(latents).to(dtype=dtype)

    if torch.any(torch.isnan(latents)):
        latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        print(f"NaN detected in latents: {cache_path}")

    return {
        "latents": latents,
        **extra_info,
    }


def check_cached_latents(latents, img_md):
    if latents is None:
        return False
    bucket_reso_hw = img_md["bucket_size"]
    latents_hw = (latents.shape[1]*8, latents.shape[2]*8)
    # assert latents_hw[0] > 128 and latents_hw[1] > 128, f"latents_hw must be larger than 128, but got latents_hw: {latents_hw}"
    # print(f"  bucket_reso_hw: {bucket_reso_hw} | latents_hw: {latents_hw}")
    return latents_hw == bucket_reso_hw


def save_latents_to_disk(latents, cache_path, original_size, crop_ltrb):
    try:
        np.savez(
            cache_path,
            latents=latents.float().cpu().numpy(),
            original_size=np.array(original_size),
            crop_ltrb=np.array(crop_ltrb),
        )
    except KeyboardInterrupt:
        raise
    if not os.path.isfile(cache_path):
        raise RuntimeError(f"Failed to save latents to {cache_path}")


async def save_latents_to_disk_async(latents_tensor, cache_path, original_size, crop_ltrb):
    import aiofiles
    from io import BytesIO
    kwargs = {}

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
    async with aiofiles.open(cache_path, 'wb') as f:
        await f.write(buffer.read())

    # 检查文件是否真的被写入
    if not os.path.isfile(cache_path):
        raise RuntimeError(f"Failed to save latents to {cache_path}")
