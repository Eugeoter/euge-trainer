from __future__ import print_function
from .utils import MODELS_DIR, DEVICE

import os
import cv2
import numpy as np
from PIL import Image
from typing import Union
from controlnet_aux import SamDetector
from controlnet_aux.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from waifuset import logging

CONTROL_TYPE = "mobile_sam"
LOGGER = logging.getLogger(CONTROL_TYPE)

MODEL_REPO_ID = "dhkim2810/MobileSAM"
DEFAULT_MODEL_NAME = "mobile_sam.pt"

SAM_DETECTOR = None


class SamDetector_Aux(SamDetector):

    def __init__(self, mask_generator: SamAutomaticMaskGenerator, sam):
        super().__init__(mask_generator)
        self.device = DEVICE
        self.model = sam.to(self.device).eval()

    @classmethod
    def from_pretrained(cls):
        """
        Possible model_type : vit_h, vit_l, vit_b, vit_t
        download weights from https://huggingface.co/dhkim2810/MobileSAM
        """
        model_dir = os.path.join(MODELS_DIR, CONTROL_TYPE)
        model_path = os.path.join(model_dir, DEFAULT_MODEL_NAME)
        if not os.path.exists(model_path):
            LOGGER.info(f"Downloading model {DEFAULT_MODEL_NAME}...")
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=DEFAULT_MODEL_NAME,
                local_dir=model_dir
            )

        sam = sam_model_registry["vit_t"](checkpoint=model_path)

        cls.model = sam.to(DEVICE).eval()

        mask_generator = SamAutomaticMaskGenerator(cls.model)

        return cls(mask_generator, sam)

    def __call__(self, input_image: Union[np.ndarray, Image.Image] = None, detect_resolution=512, image_resolution=512, output_type="cv2", **kwargs) -> np.ndarray:
        self.model.to(self.device)
        image = super().__call__(input_image=input_image, detect_resolution=detect_resolution, image_resolution=image_resolution, output_type=output_type, **kwargs)
        return np.array(image).astype(np.uint8)


def get_mobile_sam(np_img: np.ndarray, detect_resolution=512, image_resolution=512, output_type="cv2") -> np.ndarray:
    global SAM_DETECTOR
    if SAM_DETECTOR is None:
        SAM_DETECTOR = SamDetector_Aux.from_pretrained()
    seg = SAM_DETECTOR(np_img, detect_resolution=detect_resolution, image_resolution=image_resolution, output_type=output_type)
    seg = cv2.resize(seg, (np_img.shape[1], np_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return seg
