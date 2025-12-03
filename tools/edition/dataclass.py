import numpy as np
from typing import Any, Dict, List, Union
from PIL import Image


class ImageEditionData:
    def __init__(
        self,
        target_image: Union[Dict[str, Any], np.ndarray, Image.Image],
        reference_images: List[Union[Dict[str, Any], np.ndarray, Image.Image]]
    ):
        self.target_image = target_image
        self.reference_images = reference_images if isinstance(reference_images, list) else [reference_images]
