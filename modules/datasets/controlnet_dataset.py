import torch
from typing import List, Dict, Any
from waifuset.const import IMAGE_EXTS
from waifuset.classes import DictDataset
from waifuset.tools import mapping
from .t2i_dataset import T2ITrainDataset
from .mixins.controlnet_image_mixin import ControlNetImageMixin
from ..utils import dataset_utils


class ControlNetTrainDataset(T2ITrainDataset, ControlNetImageMixin):
    pass
