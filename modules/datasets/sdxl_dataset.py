import torch
from typing import List, Dict, Any
from .t2i_dataset import T2ITrainDataset


class SDXLTrainDataset(T2ITrainDataset):
    def get_size_sample(self, batch: List[str], samples: Dict[str, Any]):
        sample = dict(
            target_size_hw=[],
            original_size_hw=[],
            crop_top_lefts=[],
        )
        for i, img_key in enumerate(batch):
            img_md = self.data[img_key]
            original_size = img_md['original_size'] or img_md['image_size']
            target_size = img_md['bucket_size']
            crop_ltrb = img_md.get('crop_ltrb') or (0, 0, target_size[0], target_size[1])
            is_flipped = samples["is_flipped"][i]
            if not is_flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            sample["target_size_hw"].append((target_size[1], target_size[0]))
            sample["original_size_hw"].append((original_size[1], original_size[0]))
            sample["crop_top_lefts"].append((crop_left_top[1], crop_left_top[0]))

        sample["target_size_hw"] = torch.stack([torch.LongTensor(x) for x in sample["target_size_hw"]])
        sample["original_size_hw"] = torch.stack([torch.LongTensor(x) for x in sample["original_size_hw"]])
        sample["crop_top_lefts"] = torch.stack([torch.LongTensor(x) for x in sample["crop_top_lefts"]])
        return sample
