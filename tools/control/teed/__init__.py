from __future__ import print_function

import os
import cv2
import numpy as np
import torch
from einops import rearrange

from .ted import TED  # TEED architecture
from ..utils import MODELS_DIR, DEVICE
from waifuset import logging

CONTROL_TYPE = "teed"
LOGGER = logging.getLogger(CONTROL_TYPE)

DEFAULT_MODEL_NAME = "Annotators/7_model.pth"
DEFAULT_MODEL_REPO_ID = "bdsqlsz/qinglong_controlnet-lllite"
MTEED_MODEL_NAME = "Anyline/MTEED.pth"
MTEED_MODEL_REPO_ID = "TheMistoAI/MistoLine"

DEFAULT_DETECTOR = None
MTEED_DETECTOR = None


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


class TEEDDetector:
    """https://github.com/xavysp/TEED"""

    def __init__(self, mteed: bool = False):
        self.device = DEVICE
        self.model = TED().to(self.device).eval()

        if mteed:
            self.load_mteed_model()
        else:
            self.load_teed_model()

    def load_teed_model(self):
        """Load vanilla TEED model"""
        remote_url = os.environ.get(
            "CONTROLNET_TEED_MODEL_URL",
            "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/7_model.pth",
        )
        model_dir = os.path.join(MODELS_DIR, CONTROL_TYPE)
        model_path = os.path.join(model_dir, DEFAULT_MODEL_NAME)
        if not os.path.exists(model_path):
            from huggingface_hub import hf_hub_download
            LOGGER.info(f"Downloading model {model_path}...")
            model_path = hf_hub_download(
                repo_id=DEFAULT_MODEL_REPO_ID,
                filename=DEFAULT_MODEL_NAME,
                local_dir=model_dir
            )
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

    def load_mteed_model(self):
        """Load MTEED model for Anyline"""
        model_dir = os.path.join(MODELS_DIR, CONTROL_TYPE)
        model_path = os.path.join(model_dir, MTEED_MODEL_NAME)
        if not os.path.exists(model_path):
            from huggingface_hub import hf_hub_download
            LOGGER.info(f"Downloading model {model_path}...")
            model_path = hf_hub_download(
                repo_id=MTEED_MODEL_REPO_ID,
                filename=MTEED_MODEL_NAME,
                local_dir=model_dir
            )
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, image: np.ndarray, safe_steps: int = 2) -> np.ndarray:

        self.model.to(self.device)

        H, W, _ = image.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(image.copy()).float().to(self.device)
            image_teed = rearrange(image_teed, "h w c -> 1 c h w")
            edges = self.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [
                cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges
            ]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe_steps != 0:
                edge = safe_step(edge, safe_steps)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge


def get_teed(np_img: np.ndarray, mteed: bool = False) -> np.ndarray:
    r"""
    Get the edges of an numpy image.
    """
    global DEFAULT_DETECTOR, MTEED_DETECTOR
    if mteed:
        if MTEED_DETECTOR is None:
            MTEED_DETECTOR = TEEDDetector(mteed=True)
        edge = MTEED_DETECTOR(np_img)
    else:
        if DEFAULT_DETECTOR is None:
            DEFAULT_DETECTOR = TEEDDetector()
        edge = DEFAULT_DETECTOR(np_img)
    return edge
