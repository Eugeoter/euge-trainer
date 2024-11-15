from .controlnet_dataset import ControlNetDataset
from .sdxl_dataset import SDXLDataset


class SDXLControlNetDataset(ControlNetDataset, SDXLDataset):
    pass
