from .image_condition_dataset import ImageConditionDataset
from .sdxl_dataset import SDXLDataset


class SDXLImageConditionDataset(ImageConditionDataset, SDXLDataset):
    pass
