import torch
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from typing import List, Union
from waifuset import logging

logger = logging.get_logger("waifu_scorer")


def normalized(a: torch.Tensor, order=2, dim=-1):
    l2 = a.norm(order, dim, keepdim=True)
    l2[l2 == 0] = 1
    return a / l2


@torch.no_grad()
def encode_images(images: List[Image.Image], model2, preprocess, device='cuda', max_workers=os.cpu_count()) -> torch.Tensor:
    if isinstance(images, Image.Image):
        images = [images]

    def preprocess_single(img):
        return preprocess(img).unsqueeze(0)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers) as executor:
            image_tensors = list(executor.map(preprocess_single, images))
    else:
        image_tensors = [preprocess_single(img) for img in images]

    image_batch = torch.cat(image_tensors).to(device)
    image_features = model2.encode_image(image_batch)
    im_emb_arr = normalized(image_features).float()
    return im_emb_arr
