import numpy as np
import torch
import PIL
from typing import List, Tuple
from packaging import version
try:
    from diffusers.utils import PIL_INTERPOLATION
except ImportError:
    if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
        PIL_INTERPOLATION = {
            "linear": PIL.Image.Resampling.BILINEAR,
            "bilinear": PIL.Image.Resampling.BILINEAR,
            "bicubic": PIL.Image.Resampling.BICUBIC,
            "lanczos": PIL.Image.Resampling.LANCZOS,
            "nearest": PIL.Image.Resampling.NEAREST,
        }
    else:
        PIL_INTERPOLATION = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
            "nearest": PIL.Image.NEAREST,
        }


def prepare_controlnet_image(
    image: PIL.Image.Image,
    width: int,
    height: int,
    batch_size: int,
    num_images_per_prompt: int,
    device: torch.device,
    dtype: torch.dtype,
    do_classifier_free_guidance: bool = False,
    guess_mode: bool = False,
):
    if not isinstance(image, torch.Tensor):
        if isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            images = []

            for image_ in image:
                image_ = image_.convert("RGB")
                image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                image_ = np.array(image_)
                image_ = image_[None, :]
                images.append(image_)

            image = images

            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)

    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)

    return image


def get_sd3_pipeline(
    pretrained_model_name_or_path,
    device: torch.device,
    dtype: torch.dtype,
    include_t5: bool = False,
    cache_dir: str = None,
    token: str = None,
    xformers: str = True,
):
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
    kwargs = {}
    if not include_t5:
        kwargs.update(
            {
                "text_encoder_3": None,
                "tokenizer_3": None,
            }
        )
    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path,
        token=token,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        **kwargs,
    )
    if xformers:
        pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(device)
    return pipe
