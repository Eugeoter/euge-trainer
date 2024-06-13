import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from waifuset.classes import Data
from ...utils import dataset_utils


class CacheLatentsMixin(object):
    cache_latents: bool = True
    vae_batch_size: int = 4
    cache_to_disk: bool = True
    cache_only: bool = False
    async_cache: bool = False
    check_cache_validity: bool = True
    keep_cached_latents_in_memory: bool = True

    def load_cache_dataset(self):
        cacheset = dataset_utils.load_local_dataset(
            self.metadata_files,
            self.image_dirs,
            fp_key='cache_path',
            tbname='metadata',
            exts='.npz',
        )
        self.logger.print(f"num_caches: {len(cacheset)}")
        return cacheset

    def cache_batch_latents(self, vae, empty_cache=False):
        batches = self.make_uncached_batches()
        if not batches:
            return

        if self.accelerator.num_processes > 1:
            batches = batches[self.accelerator.process_index::self.accelerator.num_processes]  # split batches into processes
            self.logger.print(f"process {self.accelerator.process_index+1}/{self.accelerator.num_processes} | num_uncached_batches: {len(batches)}", disable=False)

        for batch in batches:
            cache_batch_latents(batch, vae, cache_to_disk=self.cache_to_disk, cache_only=self.cache_only, empty_cache=empty_cache, async_cache=self.async_cache)
        self.accelerator.wait_for_everyone()

    def make_uncached_batches(self):
        img_mds = list(self.data.values())
        img_mds.sort(key=lambda img_md: img_md['bucket_size'])
        batches = []
        batch = []
        for img_md in img_mds:
            if self.check_cache_validity:
                cache = self.get_cache(img_md['image_key'])
                latents = cache.get('latents')
                if self.keep_cached_latents_in_memory and latents is not None:
                    img_md.update(cache)
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

    def get_cache(self, img_md) -> Dict[str, Any]:
        if (latents := img_md.get('latents')) is not None:
            return {'latents': latents}
        elif (cache_path := img_md.get('cache_path')) and os.path.exists(cache_path):
            return load_latents_from_disk(cache_path, dtype=self.latents_dtype, is_main_process=self.accelerator.is_main_process)
        else:
            return {}

    def get_latents_size(self, img_md):
        if (cache_path := img_md.get('cache_path')) and os.path.exists(cache_path):
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
            print(f"remove corrupted cache file: {os.path.abspath(cache_path)}")
            print(f"  error: {e}")
        return None


def get_latents_size(npz_path):
    npz = open_cache(npz_path, mmap_mode='r')
    if npz is None:
        return None
    latents = npz["latents"]
    latents_size = latents.shape[-1], latents.shape[-2]
    return latents_size


def load_latents_from_disk(cache_path, dtype=None, mmap_mode=None, is_main_process=True) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int, int, int]]:
    cache = open_cache(cache_path, mmap_mode=mmap_mode, is_main_process=is_main_process)
    if cache is None:
        return None, None, None, None
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


def cache_batch_latents(batch: List[Data], vae, cache_to_disk, cache_only=False, async_cache=False, empty_cache=False):
    batch_images = []
    for img_md in batch:
        image = dataset_utils.load_image(img_md['image_path'])
        image = dataset_utils.process_image(image, target_size=img_md['bucket_size'])
        batch_images.append(image)

    batch_img_tensors = torch.stack(batch_images, dim=0)
    batch_img_tensors = batch_img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        batch_latents = vae.encode(batch_img_tensors).latent_dist.sample().to('cpu')

    for img_md, latents in zip(batch, batch_latents):
        # check NaN
        if torch.isnan(batch_latents).any():
            raise RuntimeError(f"NaN detected in latents: {img_md['image_path']}")

    if not async_cache:
        for img_md, latents in zip(batch, batch_latents):
            # check NaN
            if torch.isnan(batch_latents).any():
                raise RuntimeError(f"NaN detected in latents: {img_md['image_path']}")

            if cache_to_disk:
                cache_path = os.path.splitext(img_md['image_path'])[0] + ".npz"
                original_size = img_md['original_size'] or img_md['image_size']
                crop_ltrb = (0, 0, 0, 0)  # ! temporary set to 0: no crop at all
                save_latents_to_disk(latents, cache_path, original_size=original_size, crop_ltrb=crop_ltrb)
                img_md.cache_path = cache_path

            if not cache_only:
                img_md['latents'] = latents
    else:
        import asyncio

        async def save_latents():
            tasks = []
            if cache_to_disk:
                for img_md, latents in zip(batch, batch_latents):
                    cache_path = os.path.splitext(img_md['image_path'])[0] + ".npz"
                    original_size = img_md['original_size'] or img_md['image_size']
                    crop_ltrb = (0, 0, 0, 0)  # ! temporary set to 0: no crop at all
                    task = asyncio.create_task(save_latents_to_disk_async(latents, cache_path, original_size=original_size, crop_ltrb=crop_ltrb))
                    img_md['cache_path'] = cache_path
                    tasks.append(task)

            if not cache_only:
                for img_md, latents in zip(batch, batch_latents):
                    img_md['latents'] = latents

            await asyncio.gather(*tasks)
        asyncio.run(save_latents())

    # FIXME this slows down caching a lot, specify this as an option
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
