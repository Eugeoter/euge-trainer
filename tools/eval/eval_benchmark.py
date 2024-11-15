import os
import gc
import time
import torch
import cv2
import inspect
import numpy as np
from PIL import Image
from typing import Literal, List
from cleanfid import fid
from waifuset import logging
from waifuset import Dataset, FastDataset
from waifuset.utils import image_utils

logger = logging.get_logger('eval')


def eval_benchmark(
    pipeline,
    real_dataset: Dataset,  # used to generate fake images
    fake_dir,
    real_dir,
    eval_type: Literal['text-to-image', 'controllable-generation'] = 'text-to-image',
    eval_score_types: List[Literal['fid', 'clip_score', 'ssim']] = ['fid', 'clip_score'],
    max_gen=float('inf'),
    caption_getter=None,
    control_image_getter=None,
    control_scale=1.0,
    clip_batch_size=256,
):
    tic = time.time()

    # prepare save dir
    logger.info(f"Save dir: {fake_dir}")
    os.makedirs(fake_dir, exist_ok=True)

    # make batches
    logger.info(f"Number of samples: {len(real_dataset)}")

    pipeline_params = inspect.signature(pipeline.__call__).parameters

    for num_gen, (img_key, img_md) in enumerate(logging.tqdm(real_dataset.items(), desc="eval", position=0, total=min(max_gen, len(real_dataset)))):
        if num_gen >= max_gen:
            logger.info(f"Number of generations reached {max_gen}, stopping.")
            break

        save_path = os.path.join(fake_dir, f"{img_key}.png")
        if os.path.exists(save_path):
            continue

        pipeline_inputs = {
            'num_inference_steps': 28,
            'guidance_scale': 7.5,
        }

        pipeline_inputs['prompt'] = get_caption(img_md, caption_getter)

        if eval_type == 'controllable-generation':
            control_image = get_control_image(img_md, control_image_getter)

            if 'controlnet_image' in pipeline_params:
                pipeline_inputs['controlnet_image'] = control_image
            elif 'control_image' in pipeline_params:
                pipeline_inputs['control_image'] = control_image
            elif 'image' in pipeline_params:
                pipeline_inputs['image'] = control_image
            else:
                raise ValueError(f"Cannot find control image input for pipeline")

            if 'control_scale' in pipeline_params:
                pipeline_inputs['control_scale'] = control_scale
            elif 'controlnet_scale' in pipeline_params:
                pipeline_inputs['controlnet_scale'] = control_scale
            pipeline_inputs['width'] = control_image.width
            pipeline_inputs['height'] = control_image.height
        else:
            pipeline_inputs['width'] = 512
            pipeline_inputs['height'] = 512

        with torch.no_grad():
            image = pipeline.__call__(**pipeline_inputs).images[0]

        # save images
        image.save(save_path)

    else:
        logger.info(f"Run out of validation data, stopping.")

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    toc = time.time()
    logger.print(f"Generation done. Time cost: {toc - tic:.2f}s")

    if 'fid' in eval_score_types:
        logger.info(f"Calculating FID score...")
        fid_score = fid.compute_fid(real_dir, fake_dir, mode='clean')
        logger.print(f"FID Score: {fid_score}")

    if 'clip_score' in eval_score_types:
        logger.info(f"Calculating CLIP score...")
        fake_dataset = real_dataset.copy()
        fake_dataset.redirect(['image_path'], FastDataset(fake_dir, recur=False, verbose=True))
        fake_dataset = fake_dataset.subset(lambda x: os.path.exists(x['image_path']))
        clip_score = compute_clip_score(fake_dataset, caption_getter=caption_getter, batch_size=clip_batch_size)
        logger.print(f"CLIP Score: {clip_score}")

    if 'ssim' in eval_score_types:
        logger.info(f"Calculating SSIM score...")
        ssim_score = compute_ssim(real_dataset, FastDataset(fake_dir, recur=False, verbose=True))
        logger.print(f"SSIM Score: {ssim_score}")

    return {'fid': fid_score, 'clip_score': clip_score}


def get_caption(img_md, getter=None):
    if getter is not None:
        return getter(img_md)
    elif 'caption' in img_md:
        return img_md['caption']
    else:
        raise ValueError(f"Cannot find caption for {img_md}")


def get_control_image(img_md, getter=None):
    img_key = img_md['image_key']
    if getter is not None:
        control_image = getter(img_md)
    elif 'control_image' in img_md:
        control_image = img_md['control_image']
    elif 'control_image_path' in img_md:
        control_image = Image.open(img_md['control_image_path']).convert('RGB')
    else:
        raise ValueError(f"Cannot find control image for {img_key}: {img_md}")

    if isinstance(control_image, np.ndarray):
        control_image = Image.fromarray(control_image)

    width, height = control_image.size
    width = width // 8 * 8
    height = height // 8 * 8
    control_image = image_utils.resize_if_needed(control_image, (width, height))

    return control_image


@torch.no_grad()
def compute_clip_score(
    dataset: Dataset,
    caption_getter=None,
    batch_size=64,
    device=None,
):
    import clip
    chunks = dataset.chunks(batch_size)
    model, preprocess = clip.load("ViT-B/32", device=device)

    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()

    for chunk in logger.tqdm(chunks, desc='computing clip score'):
        images = torch.stack([preprocess(Image.open(img_md['image_path'])) for img_md in chunk.values()], dim=0).to(device)
        image_features = model.encode_image(images)

        captions = [get_caption(img_md, caption_getter) for img_md in chunk.values()]
        input_ids = clip.tokenize(captions, truncate=True).to(device)
        text_features = model.encode_text(input_ids)

        # normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True).to(torch.float32)
        text_features = text_features / text_features.norm(dim=1, keepdim=True).to(torch.float32)

        # calculate scores
        score = logit_scale * (image_features * text_features).sum()
        score_acc += score

        sample_num += len(chunk)

    return score_acc / sample_num


def compute_ssim(
    dataset_1: Dataset,
    dataset_2: Dataset,
):
    from skimage.metrics import structural_similarity as ssim

    intersection = set(dataset_1.keys()) & set(dataset_2.keys())
    dataset_1 = dataset_1.subset(lambda x: x['image_key'] in intersection)
    dataset_2 = dataset_2.subset(lambda x: x['image_key'] in intersection)

    ssim_acc = 0.
    sample_num = 0.

    for img_key in logger.tqdm(dataset_1.keys(), desc='computing ssim score'):
        img_md_1 = dataset_1[img_key]
        img_md_2 = dataset_2[img_key]
        img_1 = cv2.imread(img_md_1['image_path'])
        img_2 = cv2.imread(img_md_2['image_path'])
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        ssim_acc += ssim(img_1, img_2, multichannel=False)
        sample_num += 1

    return ssim_acc / sample_num
