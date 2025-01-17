from .hunyuan_dataset import HunyuanDataset
from .image_condition_dataset import ImageConditionDataset


class HunyuanControlNetDataset(HunyuanDataset, ImageConditionDataset):
    pass
