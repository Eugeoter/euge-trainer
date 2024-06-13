from .controlnet_dataset import ControlNetTrainDataset
from .sdxl_dataset import SDXLTrainDataset


class SDXLControlNetTrainDataset(ControlNetTrainDataset, SDXLTrainDataset):
    pass
