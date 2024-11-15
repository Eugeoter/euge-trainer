import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Any
from waifuset import Data, HuggingFaceData
from waifuset import logging
from ...utils import ws_train_utils

logger = logging.get_logger('dataset')


class CacheImageEmbeddingMixin(object):
    flip_aug: bool = True

    cache_dir: str = None
    cache_column: str = 'cache'
    cache_flipped_column: str = 'cache_flipped'
    cache_path_column: str = 'cache_path'
    cache_batch_size: int = 512

    cache_img_embs: bool = False
    cache_to_disk: bool = True
    cache_only: bool = False
    async_cache: bool = False
    check_cache_validity: bool = True
    cache_img_emb_max_dataloader_n_workers: int = 4
    keep_cached_img_emb_in_memory: bool = True

    def _print_cache_img_emb_message(self):
        self.logger.print(logging.green(f"==================== START CACHING ===================="))
        self.logger.print(f"cache_img_emb: {self.cache_img_embs} | cache_to_disk: {self.cache_to_disk} | cache_only: {self.cache_only}")
        self.logger.print(f"async_cache: {self.async_cache} | check_cache_validity: {self.check_cache_validity}")
        self.logger.print(f"cache_img_emb_max_dataloader_n_workers: {self.cache_img_emb_max_dataloader_n_workers} | keep_cached_img_emb_in_memory: {self.keep_cached_img_emb_in_memory}")
        self.logger.print(f"cache_batch_size: {self.cache_batch_size}")

    def cache_batch_image_embeddings(self, clip_model, clip_preprocessor, empty_cache=False):
        batches = self.make_cache_batches()
        if not batches:
            self.logger.info(f"all image embeddings are cached")
            return
        if self.accelerator.num_processes > 1:
            batches = batches[self.accelerator.process_index::self.accelerator.num_processes]  # split batches into processes
            self.logger.print(f"process {self.accelerator.process_index+1}/{self.accelerator.num_processes} | num_uncached_batches: {len(batches)}", disable=False)

        class CacheImageEmbeddingDataset(Dataset):
            def __init__(self, batches, dataset, img_getter):
                self.batches = batches
                self.dataset = dataset
                self.img_getter = img_getter

            def __len__(self):
                return len(self.batches)

            def __getitem__(self, idx):
                batch = self.batches[idx]
                sample = dict(
                    images=[],
                    image_keys=batch,
                )
                for img_key in batch:
                    img_md = self.dataset[img_key]
                    image = self.img_getter(img_md)
                    sample['images'].append(image)
                return sample

        if batches:
            dataset = CacheImageEmbeddingDataset(batches, self.dataset, img_getter=self.get_image)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=self.cache_img_emb_max_dataloader_n_workers,
                pin_memory=True,
                collate_fn=self.collate_fn,
                shuffle=True,
            )

            os.makedirs(self.cache_dir, exist_ok=True)
            for sample in self.logger.tqdm(dataloader, total=len(dataloader), desc="caching image embeddings"):
                self.cache_sample_image_embeddings(sample, clip_model, clip_preprocessor, empty_cache=empty_cache)
        self.accelerator.wait_for_everyone()

    def cache_sample_image_embeddings(self, sample, clip_model, clip_preprocessor, empty_cache=False):
        batch_img_keys = sample['image_keys']
        batch_images: List[Image.Image] = sample['images']

        with torch.no_grad():
            batch_img_embs = ws_train_utils.encode_images(
                batch_images,
                clip_model,
                clip_preprocessor,
                self.accelerator.device,
                max_workers=self.cache_img_emb_max_dataloader_n_workers
            ).cpu()
            if self.flip_aug:
                batch_images_flipped = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in batch_images]
                batch_img_embs_flipped = ws_train_utils.encode_images(
                    batch_images_flipped,
                    clip_model, clip_preprocessor,
                    self.accelerator.device,
                    max_workers=self.cache_img_emb_max_dataloader_n_workers
                ).cpu()

        if not self.async_cache:
            for i, (img_key, img_emb) in enumerate(zip(batch_img_keys, batch_img_embs)):
                img_md = self.get_img_md(img_key)
                if self.cache_to_disk:
                    cache_path = self.get_cache_path(img_md)
                    save_img_emb_to_disk(
                        img_emb,
                        cache_path,
                        img_emb_flipped=batch_img_embs_flipped[i] if self.flip_aug else None,
                    )
                    img_md[self.cache_path_column] = cache_path

                if not self.cache_only:
                    img_md[self.cache_column] = img_emb
                    img_md[self.cache_flipped_column] = batch_img_embs_flipped[i] if self.flip_aug else None
        else:
            import asyncio

            async def save_img_emb():
                tasks = []
                if self.cache_to_disk:
                    for i, (img_key, img_emb) in enumerate(zip(batch_img_keys, batch_img_embs)):
                        img_md = self.get_img_md(img_key)
                        cache_path = self.get_cache_path(img_md)
                        task = asyncio.create_task(save_img_emb_to_disk_async(
                            img_emb,
                            cache_path,
                            img_emb_flipped=batch_img_embs_flipped[i] if self.flip_aug else None,
                        ))
                        img_md[self.cache_path_column] = cache_path
                        tasks.append(task)

                if not self.cache_only:
                    for i, (img_key, img_emb) in enumerate(zip(batch_img_keys, batch_img_embs)):
                        img_md = self.get_img_md(img_key)
                        img_md[self.cache_column] = img_emb
                        img_md[self.cache_flipped_column] = batch_img_embs_flipped[i] if self.flip_aug else None

                await asyncio.gather(*tasks)
            asyncio.run(save_img_emb())

        # FIXME this slows down caching a lot, specify this as an option
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_cache_supported(self, img_md):
        return not isinstance(img_md, HuggingFaceData)

    def get_cache_path(self, img_md):
        return img_md.get(self.cache_path_column) or os.path.join(self.cache_dir, os.path.basename(os.path.splitext(img_md['image_path'])[0]) + '.npz')

    def make_cache_batches(self) -> List[Data]:
        img_keys = list(self.dataset.keys())
        batches = []
        batch = []
        for img_key in img_keys:
            img_md = self.dataset[img_key]
            if self.check_cache_validity:
                if not self.is_cache_supported(img_md):
                    continue
                cache = self.get_cache(img_md, update=self.keep_cached_img_emb_in_memory)
                img_emb = cache.get(self.cache_column)
                img_emb_flipped = cache.get(self.cache_flipped_column)
                if img_emb is not None and (not self.flip_aug or img_emb_flipped is not None):
                    continue
            if os.path.exists(img_md.get('cache_path', '')):
                continue
            batch.append(img_key)
            if len(batch) >= self.cache_batch_size:
                batches.append(batch)
                batch = []
        if len(batch) > 0:
            batches.append(batch)
        return batches

    def get_cache(self, img_md, update=False) -> Dict[str, Any]:
        if not self.cache_img_embs:
            return {}

        elif (img_emb := img_md.get(self.cache_column)) is not None or (self.flip_aug and img_md.get(self.cache_flipped_column) is not None):
            cache = {}
            if img_emb is not None:
                cache[self.cache_column] = img_emb
            if self.flip_aug and (img_emb_flipped := img_md.get(self.cache_flipped_column)) is not None:
                cache[self.cache_flipped_column] = img_emb_flipped

        elif (cache_path := self.get_cache_path(img_md)) and os.path.exists(cache_path):
            raw_cache = load_img_emb_from_disk(cache_path, dtype=None, is_main_process=self.accelerator.is_main_process)
            cache = {}
            if (img_emb := raw_cache.get('emb')) is not None:
                cache[self.cache_column] = img_emb
            if (img_emb_flipped := raw_cache.get('emb_flipped')) is not None:
                cache[self.cache_flipped_column] = img_emb_flipped

        else:
            cache = {}

        if update:
            img_md.update(cache)

        return cache


def open_cache(cache_path, mmap_mode=None, is_main_process=True):
    try:
        cache = np.load(cache_path, mmap_mode=mmap_mode)
        return cache
    except Exception as e:
        if is_main_process:
            import shutil
            backup_path = str(cache_path) + '.bak'
            shutil.move(str(cache_path), backup_path)
            logger.error(f"remove corrupted cache file: {os.path.abspath(cache_path)}, error: {e}")
        return None


def load_img_emb_from_disk(cache_path, dtype=None, mmap_mode=None, is_main_process=True) -> Dict[str, Any]:
    cache = open_cache(cache_path, mmap_mode=mmap_mode, is_main_process=is_main_process)
    if cache is None:
        return {}
    img_emb = cache["emb"]
    img_emb = torch.FloatTensor(img_emb).to(dtype=dtype)

    img_emb_flipped = cache.get("emb_flipped", None)
    if img_emb_flipped is not None:
        img_emb_flipped = torch.FloatTensor(img_emb_flipped).to(dtype=dtype)

    # if torch.any(torch.isnan(img_emb)):
    #     img_emb = torch.where(torch.isnan(img_emb), torch.zeros_like(img_emb), img_emb)
    #     logger.warning(f"NaN detected in image embedding cache file: {cache_path}")

    return {"emb": img_emb, "emb_flipped": img_emb_flipped}


def save_img_emb_to_disk(img_emb, cache_path, img_emb_flipped=None):
    try:
        extra_kwargs = {}
        if img_emb_flipped is not None:
            extra_kwargs.update(emb_flipped=img_emb_flipped.float().cpu().numpy())
        np.savez(
            cache_path,
            emb=img_emb.float().cpu().numpy(),
            **extra_kwargs,
        )
    except KeyboardInterrupt:
        raise
    if not os.path.isfile(cache_path):
        raise RuntimeError(f"Failed to save image embedding to {cache_path}")


async def save_img_emb_to_disk_async(img_emb, cache_path, img_emb_flipped=None):
    import aiofiles
    from io import BytesIO
    # 使用BytesIO作为临时内存文件
    extra_kwargs = {}
    if img_emb_flipped is not None:
        extra_kwargs.update(emb_flipped=img_emb_flipped.float().cpu().numpy())
    buffer = BytesIO()
    np.savez(
        buffer,
        emb=img_emb.float().cpu().numpy(),
        **extra_kwargs,
    )

    # 将BytesIO缓冲区的内容异步写入磁盘
    buffer.seek(0)  # 重置buffer的位置到开始
    async with aiofiles.open(cache_path, 'wb') as f:
        await f.write(buffer.read())

    # 检查文件是否真的被写入
    if not os.path.isfile(cache_path):
        raise RuntimeError(f"Failed to save image embedding to {cache_path}")
