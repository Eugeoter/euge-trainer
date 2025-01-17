import numpy as np
from PIL import Image
from waifuset import logging

LOGGER = logging.get_logger("inpaint")
MASK_GENERATOR = None


def get_random_inpaint_mask(
    img,
    box_proba=1/2,
    bbox_min_size=30,
    bbox_max_size=400,
    bbox_min_times=0,
    bbox_max_times=3,
    irregular_proba=1/2,
    irregular_max_angle=6,
    irregular_max_len=240,
    irregular_max_width=80,
    irregular_min_times=0,
    irregular_max_times=3,
    superres_proba=0,
    segm_proba=0,
    variants_n=1,
) -> Image.Image:
    global MASK_GENERATOR
    if MASK_GENERATOR is None:
        from .masks import MakeManyMasksWrapper, MixedMaskGenerator
        box_kwargs = dict(
            bbox_min_size=bbox_min_size,
            bbox_max_size=bbox_max_size,
            min_times=bbox_min_times,
            max_times=bbox_max_times,
        )
        irregular_kwargs = dict(
            max_angle=irregular_max_angle,
            max_len=irregular_max_len,
            max_width=irregular_max_width,
            min_times=irregular_min_times,
            max_times=irregular_max_times,
        )
        MASK_GENERATOR = MakeManyMasksWrapper(
            MixedMaskGenerator(
                irregular_proba=irregular_proba,
                irregular_kwargs=irregular_kwargs,
                box_proba=box_proba,
                box_kwargs=box_kwargs,
                superres_proba=superres_proba,
                segm_proba=segm_proba,
            ),
            variants_n=variants_n
        )
    src_masks = MASK_GENERATOR.get_masks(img)
    max_tamper_area = 1
    max_masks_per_image = 100
    filtered_image_mask_pairs = []
    for cur_mask in src_masks:
        cur_image = img
        if len(np.unique(cur_mask)) <= 1 or cur_mask.mean() > max_tamper_area:
            continue
        filtered_image_mask_pairs.append((cur_image, cur_mask))
    mask_indices = np.random.choice(
        len(filtered_image_mask_pairs),
        size=min(len(filtered_image_mask_pairs), max_masks_per_image),
        replace=False
    )
    # return Image.fromarray(
    #     np.clip(
    #         filtered_image_mask_pairs[mask_indices[0]][1] * 255,
    #         0,
    #         255
    #     ).astype('uint8'),
    #     mode='L'
    # )

    total_mask = Image.new('L', img.size, 0)

    for i, idx in enumerate(mask_indices):
        cur_image, cur_mask = filtered_image_mask_pairs[idx]
        cur_mask = Image.fromarray(
            np.clip(
                cur_mask * 255,
                0,
                255
            ).astype('uint8'),
            mode='L'
        )
        total_mask = Image.composite(
            cur_mask,
            total_mask,
            cur_mask
        )

    return total_mask
