import random
import numpy as np
import imagesize


def get_coco2017_caption(img_md, dataset_info=None, **kwargs) -> str:
    img_id = img_md['img_id']
    coco = img_md['cocos']['captions']
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    caption = random.choice(anns)['caption']
    return caption


def get_coco2017_segm(img_md, dataset_info=None, **kwargs) -> np.ndarray:
    img_id = img_md['img_id']
    coco = img_md['cocos']['instances']
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=img_md['cat_ids'])
    anns = coco.loadAnns(ann_ids)
    width, height = imagesize.get(img_md['image_path'])
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        mask += coco.annToMask(ann)
    mask = mask.astype(np.uint8) * 255
    return mask
