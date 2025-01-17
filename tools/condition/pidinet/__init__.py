import os
import torch
import numpy as np
import safetensors
from einops import rearrange
from .model import pidinet
from waifuset import logging

from ..utils import MODELS_DIR, DEVICE

CONTROL_TYPE = "pidinet"
LOGGER = logging.getLogger(CONTROL_TYPE)
MODEL = None
MODEL_REPO_ID = "lllyasviel/Annotators"
DEFAULT_MODEL_NAME = "table5_pidinet.pth"


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = torch.load(ckpt_path, map_location=torch.device(location))
    state_dict = get_state_dict(state_dict)
    LOGGER.info(f"Loaded state_dict from [{ckpt_path}]")
    return state_dict


def get_state_dict(d):
    return d.get("state_dict", d)


def apply_pidinet(input_image, is_safe=False, apply_fliter=False):
    global MODEL
    if MODEL is None:
        model_dir = os.path.join(MODELS_DIR, CONTROL_TYPE)
        model_path = os.path.join(model_dir, "table5_pidinet.pth")
        if not os.path.exists(model_path):
            from huggingface_hub import hf_hub_download
            print(f"Downloading model {model_path}...")
            model_path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=DEFAULT_MODEL_NAME,
                local_dir=model_dir
            )
        MODEL = pidinet()
        ckp = load_state_dict(model_path)
        MODEL.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()})

    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    assert input_image.ndim == 3
    input_image = input_image[:, :, ::-1].copy()
    with torch.no_grad():
        image_pidi = torch.from_numpy(input_image).float().to(DEVICE)
        image_pidi = image_pidi / 255.0
        image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
        edge = MODEL(image_pidi)[-1]
        edge = edge.cpu().numpy()
        if apply_fliter:
            edge = edge > 0.5
        if is_safe:
            edge = safe_step(edge)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

    return edge[0][0]


def unload_pid_model():
    global MODEL
    if MODEL is not None:
        MODEL.cpu()


def get_pidinet(np_img: np.ndarray, is_safe=False, apply_fliter=False) -> np.ndarray:
    r"""
    Get the scribble of an numpy image.
    """
    return apply_pidinet(np_img, is_safe, apply_fliter)
