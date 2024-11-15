import cv2
import numpy as np
import torch
import os
from huggingface_hub import hf_hub_download
from .models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines
from waifuset import logging
from ..utils import MODELS_DIR, DEVICE

CONTROL_TYPE = "mlsd"
LOGGER = logging.getLogger(CONTROL_TYPE)
MLSD_MODEL = None
MODEL_REPO_ID = "lllyasviel/Annotators"
DEFAULT_MODEL_NAME = "mlsd_large_512_fp32.pth"


def unload_mlsd_model():
    global MLSD_MODEL
    if MLSD_MODEL is not None:
        MLSD_MODEL = MLSD_MODEL.cpu()


def apply_mlsd(input_image, thr_v, thr_d, device=DEVICE) -> np.ndarray:
    global MLSD_MODEL
    if MLSD_MODEL is None:
        model_dir = os.path.join(MODELS_DIR, CONTROL_TYPE)
        model_path = os.path.join(model_dir, DEFAULT_MODEL_NAME)
        if not os.path.exists(model_path):
            LOGGER.info(f"Downloading {DEFAULT_MODEL_NAME} from {MODEL_REPO_ID}")
            model_path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=DEFAULT_MODEL_NAME,
                local_dir=model_dir
            )
        MLSD_MODEL = MobileV2_MLSD_Large()
        MLSD_MODEL.load_state_dict(torch.load(model_path), strict=True)
    MLSD_MODEL = MLSD_MODEL.to(device).eval()

    model = MLSD_MODEL
    assert input_image.ndim == 3
    img = input_image
    img_output = np.zeros_like(img)
    try:
        with torch.no_grad():
            lines = pred_lines(img, model, [img.shape[0], img.shape[1]], thr_v, thr_d, device=device)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
    except Exception as e:
        pass
    return img_output[:, :, 0]


def get_mlsd(np_img: np.ndarray, thr_v: float = 0.5, thr_d: float = 0.5, device=DEVICE) -> np.ndarray:
    return apply_mlsd(np_img, thr_v, thr_d, device=DEVICE)
