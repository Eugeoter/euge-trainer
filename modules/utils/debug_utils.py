import numpy as np
import torch
import warnings
from PIL import Image
from typing import Union


def ignore_warnings(categories=[FutureWarning, DeprecationWarning]):
    for category in categories:
        warnings.filterwarnings("ignore", category=category)
