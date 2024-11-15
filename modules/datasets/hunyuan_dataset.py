import torch
import numpy as np
from typing import List, Dict, Any, Literal, Union, Tuple
from .t2i_dataset import T2IDataset
from ..models.hunyuan.modules.posemb_layers import get_2d_rotary_pos_embed, calc_sizes


class HunyuanDataset(T2IDataset):
    size_cond: Union[int, Tuple[int, int]] = None
    use_style_cond: bool = False
    rope_img: Literal['extend', 'base512', 'base1024'] = 'base1024'
    rope_real: bool = True
    freqs_cis_img: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    cache_posemb: bool = True

    patch_size: int
    hidden_size: int
    num_heads: int

    # def cache_image_posemb(
    #     self,
    #     patch_size: int,
    #     hidden_size: int,
    #     num_heads: int,
    # ):
    #     self.logger.info(f"caching positional embeddings...")
    #     freqs_cis_img = {}
    #     resolutions = self.buckets.keys()

    #     pbar = self.logger.tqdm(total=len(resolutions), desc="cache posemb")
    #     logs = {'resolution': None}
    #     for reso in resolutions:
    #         th, tw = reso[1] // 8 // patch_size, reso[0] // 8 // patch_size
    #         sub_args = calc_sizes(self.rope_img, patch_size, th, tw)
    #         freqs_cis_img[reso] = get_2d_rotary_pos_embed(hidden_size // num_heads, *sub_args, use_real=self.rope_real)
    #         # self.logger.info(f"    Using image RoPE ({self.rope_img}) ({'real' if self.rope_real else 'complex'}): {sub_args} | ({reso}) "
    #         #                  f"{freqs_cis_img[reso][0].shape if self.rope_real else freqs_cis_img[reso].shape}")
    #         logs['resolution'] = f"{reso[0]}x{reso[1]}"
    #         pbar.set_postfix(logs)
    #         pbar.update(1)
    #     self.freqs_cis_img = freqs_cis_img

    def get_freqs_cis_img(self, size: Tuple[int, int]):
        if self.cache_posemb and size in self.freqs_cis_img:
            return self.freqs_cis_img[size]
        patch_size, hidden_size, num_heads = self.patch_size, self.hidden_size, self.num_heads
        th, tw = size[1] // 8 // patch_size, size[0] // 8 // patch_size
        sub_args = calc_sizes(self.rope_img, patch_size, th, tw)
        freqs_cis_img = get_2d_rotary_pos_embed(hidden_size // num_heads, *sub_args, use_real=self.rope_real)
        if self.cache_posemb:
            self.freqs_cis_img[size] = freqs_cis_img
        return freqs_cis_img

    def get_size_sample(self, batch: List[str], samples: Dict[str, Any]):
        sample = dict(
            image_meta_size=[],
        )
        for i, img_key in enumerate(batch):
            img_md = self.dataset[img_key]
            image_size, original_size, bucket_size = self.get_size(img_md)
            original_size = original_size or image_size
            target_size = bucket_size
            crop_ltrb = img_md.get('crop_ltrb') or (0, 0, target_size[0], target_size[1])
            # crop_ltrb = (0, 0, 0, 0)
            is_flipped = samples["is_flipped"][i]
            if not is_flipped:
                crop_top_left = (crop_ltrb[1], crop_ltrb[0])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_top_left = (crop_ltrb[1], target_size[0] - crop_ltrb[2])
            image_meta_size = tuple(original_size) + tuple(bucket_size) + tuple(crop_top_left)
            image_meta_size = torch.tensor(np.array(image_meta_size)).clone().detach()
            sample["image_meta_size"].append(image_meta_size)
        sample['image_meta_size'] = torch.stack(sample['image_meta_size'])
        return sample

    def get_style_sample(self, batch: List[str], samples: Dict[str, Any]):
        sample = dict(
            style=[],
        )
        for i, img_key in enumerate(batch):
            img_md = self.dataset[img_key]
            style = img_md.get('style_index', 0)
            style = torch.tensor(style).clone().detach()
            sample["style"].append(style)
        sample['style'] = torch.stack(sample['style'])
        return sample

    def get_positional_embedding_sample(self, batch: List[str], samples: Dict[str, Any]):
        sample = dict(
            cos_cis_img=[],
            sin_cis_img=[],
        )
        img_md = self.dataset[batch[0]]
        _, _, bucket_size = self.get_size(img_md)
        cos_cis_img, sin_cis_img = self.get_freqs_cis_img(bucket_size)
        sample['cos_cis_img'] = cos_cis_img
        sample['sin_cis_img'] = sin_cis_img
        return sample
