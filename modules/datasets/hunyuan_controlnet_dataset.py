from .hunyuan_dataset import HunyuanDataset
from .controlnet_dataset import ControlNetDataset


class HunyuanControlNetDataset(HunyuanDataset, ControlNetDataset):
    pass
