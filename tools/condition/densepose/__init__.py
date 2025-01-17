import torchvision  # Fix issue Unknown builtin op: torchvision::nms
import cv2
import numpy as np
import torch
from einops import rearrange
from .densepose import DensePoseMaskedColormapResultsVisualizer, _extract_i_from_iuvarr, densepose_chart_predictor_output_to_result_with_confidences
import os
from waifuset import logging
from ..utils import MODELS_DIR, DEVICE

CONTROL_TYPE = "densepose"
LOGGER = logging.get_logger(CONTROL_TYPE)
MODEL_REPO_ID = "LayerNorm/DensePose-TorchScript-with-hint-image"
DEFAULT_MODEL_NAME = "densepose_r50_fpn_dl.torchscript"

N_PART_LABELS = 24
RESULT_VISUALIZER = DensePoseMaskedColormapResultsVisualizer(
    alpha=1,
    data_extractor=_extract_i_from_iuvarr,
    segm_extractor=_extract_i_from_iuvarr,
    val_scale=255.0 / N_PART_LABELS
)
TORCHSCRIPT_MODEL = None


def apply_densepose(input_image, cmap="viridis"):
    global TORCHSCRIPT_MODEL
    if TORCHSCRIPT_MODEL is None:
        model_dir = os.path.join(MODELS_DIR, "densepose")
        model_path = os.path.join(model_dir, "densepose_r50_fpn_dl.torchscript")
        if not os.path.exists(model_path):
            LOGGER.info(f"Downloading model {DEFAULT_MODEL_NAME}...")
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=DEFAULT_MODEL_NAME,
                local_dir=model_dir
            )
        TORCHSCRIPT_MODEL = torch.jit.load(model_path, map_location="cpu").to(DEVICE).eval()
    H, W = input_image.shape[:2]

    hint_image_canvas = np.zeros([H, W], dtype=np.uint8)
    hint_image_canvas = np.tile(hint_image_canvas[:, :, np.newaxis], [1, 1, 3])
    input_image = rearrange(torch.from_numpy(input_image).to(DEVICE), 'h w c -> c h w')
    pred_boxes, corase_segm, fine_segm, u, v = TORCHSCRIPT_MODEL(input_image)

    extractor = densepose_chart_predictor_output_to_result_with_confidences
    densepose_results = [extractor(pred_boxes[i:i+1], corase_segm[i:i+1], fine_segm[i:i+1], u[i:i+1], v[i:i+1]) for i in range(len(pred_boxes))]

    if cmap == "viridis":
        RESULT_VISUALIZER.mask_visualizer.cmap = cv2.COLORMAP_VIRIDIS
        hint_image = RESULT_VISUALIZER.visualize(hint_image_canvas, densepose_results)
        hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)
        hint_image[:, :, 0][hint_image[:, :, 0] == 0] = 68
        hint_image[:, :, 1][hint_image[:, :, 1] == 0] = 1
        hint_image[:, :, 2][hint_image[:, :, 2] == 0] = 84
    else:
        RESULT_VISUALIZER.mask_visualizer.cmap = cv2.COLORMAP_PARULA
        hint_image = RESULT_VISUALIZER.visualize(hint_image_canvas, densepose_results)
        hint_image = cv2.cvtColor(hint_image, cv2.COLOR_BGR2RGB)

    return hint_image


def unload_model():
    global TORCHSCRIPT_MODEL
    if TORCHSCRIPT_MODEL is not None:
        TORCHSCRIPT_MODEL.cpu()


def get_densepose(np_img: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    r"""
    Get the densepose of an numpy image.
    """
    return apply_densepose(np_img, cmap)
