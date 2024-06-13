import torch
from typing import List, Dict, Any
from waifuset.const import IMAGE_EXTS
from waifuset.classes import DictDataset
from waifuset.tools import mapping
from .t2i_dataset import T2ITrainDataset
from .mixins.controlnet_image_mixin import ControlNetImageMixin
from ..utils import dataset_utils


class ControlNetTrainDataset(T2ITrainDataset, ControlNetImageMixin):
    control_image_dirs: List[str] = None
    dataset_control_image_column: str = 'conditioning_image'

    def load_local_dataset(self) -> DictDataset:
        imageset = super().load_local_dataset()
        if self.control_image_dirs is not None:
            controlset = self.load_control_image_dataset()
            imageset.apply_map(mapping.redirect_columns, columns=['control_image_path'], tarset=controlset)
        return imageset

    def load_control_image_dataset(self) -> DictDataset:
        control_imageset = dataset_utils.load_local_dataset(
            self.metadata_files,
            self.control_image_dirs,
            tbname='metadata',
            primary_key='control_image_key',
            fp_key='control_image_path',
            exts=IMAGE_EXTS,
        )
        self.logger.print(f"num_control_images: {len(control_imageset)}")
        return control_imageset

    def get_hf_dataset_column_mapping(self):
        column_mapping = super().get_hf_dataset_column_mapping()
        column_mapping[self.dataset_control_image_column] = 'control_image'
        return column_mapping

    def get_control_image_sample(self, batch: List[str], samples: Dict[str, Any]) -> Dict[str, Any]:
        sample = dict(
            control_images=[],
        )
        for i, img_key in enumerate(batch):
            img_md = self.data[img_key]
            control_image = self.get_control_image(img_md)
            sample["control_images"].append(control_image)
        sample["control_images"] = torch.stack(sample["control_images"], dim=0).to(memory_format=torch.contiguous_format).float()
        return sample
