import math
import random
from PIL import Image, ExifTags
from typing import List, Dict, Optional, Tuple, Union
from ...utils import log_utils, dataset_utils

SDXL_BUCKET_SIZES = [
    (512, 1856), (512, 1920), (512, 1984), (512, 2048),
    (576, 1664), (576, 1728), (576, 1792), (640, 1536),
    (640, 1600), (704, 1344), (704, 1408), (704, 1472),
    (768, 1280), (768, 1344), (832, 1152), (832, 1216),
    (896, 1088), (896, 1152), (960, 1024), (960, 1088),
    (1024, 960), (1024, 1024), (1088, 896), (1088, 960),
    (1152, 832), (1152, 896), (1216, 832), (1280, 768),
    (1344, 704), (1344, 768), (1408, 704), (1472, 704),
    (1536, 640), (1600, 640), (1664, 576), (1728, 576),
    (1792, 576), (1856, 512), (1920, 512), (1984, 512), (2048, 512)
]


class AspectRatioBucketMixin(object):
    resolution: Union[Tuple[int, int], int]
    arb: bool = True
    max_aspect_ratio: float = 1.1
    bucket_reso_step: int = 32
    predefined_buckets: Optional[List[Tuple[int, int]]] = []

    def get_resolution(self):
        return self.resolution if isinstance(self.resolution, tuple) else (self.resolution, self.resolution)

    def get_bucket_size(self, img_md, image_size=None):
        if not self.arb:
            return self.get_resolution()
        elif (bucket_size := img_md.get('bucket_size')):
            return bucket_size
        elif image_size is not None or (image_size := img_md.get('image_size')) is not None:
            image_size = dataset_utils.convert_size_if_needed(image_size)
            bucket_size = get_bucket_reso(
                image_size,
                self.resolution,
                max_aspect_ratio=self.max_aspect_ratio,
                divisible=self.bucket_reso_step,
                buckets=self.predefined_buckets,
            )
            return bucket_size
        else:
            raise ValueError("Either `bucket_size` or `image_size` must be provided in metadata.")

    def make_buckets(self) -> Dict[Tuple[int, int], List[str]]:
        if not self.arb:
            return {self.get_resolution(): list(self.data.keys())}
        bucket_keys = {}
        for img_key, img_md in self.data.items():
            if (bucket_size := img_md.get('bucket_size')) is not None:
                pass
            else:
                bucket_size = self.get_bucket_size(img_md)
            bucket_keys.setdefault(bucket_size, []).extend([img_key] * img_md.get('weight', 1))
        bucket_keys = self.shuffle_buckets(bucket_keys)
        return bucket_keys

    def shuffle_buckets(self, buckets: Dict[Tuple[int, int], List[str]]):
        bucket_sizes = list(buckets.keys())
        random.shuffle(bucket_sizes)
        buckets = {k: buckets[k] for k in bucket_sizes}
        for bucket in buckets.values():
            random.shuffle(bucket)
        return buckets


def around_reso(img_w, img_h, reso: Union[Tuple[int, int], int], divisible: Optional[int] = None) -> Tuple[int, int]:
    r"""
    w*h = reso*reso
    w/h = img_w/img_h
    => w = img_ar*h
    => img_ar*h^2 = reso
    => h = sqrt(reso / img_ar)
    """
    reso = reso if isinstance(reso, tuple) else (reso, reso)
    divisible = divisible or 1
    img_ar = img_w / img_h
    around_h = int(math.sqrt(reso[0]*reso[1] /
                   img_ar) // divisible * divisible)
    around_w = int(img_ar * around_h // divisible * divisible)
    return (around_w, around_h)


def aspect_ratio_diff(size_1: Tuple[int, int], size_2: Tuple[int, int]):
    ar_1 = size_1[0] / size_1[1]
    ar_2 = size_2[0] / size_2[1]
    return max(ar_1/ar_2, ar_2/ar_1)


def rotate_image_straight(image: Image) -> Image:
    exif: Image.Exif = image.getexif()
    if exif:
        orientation_tag = {v: k for k, v in ExifTags.TAGS.items()}[
            'Orientation']
        orientation = exif.get(orientation_tag)
        degree = {
            3: 180,
            6: 270,
            8: 90,
        }.get(orientation)
        if degree:
            image = image.rotate(degree, expand=True)
    return image


def closest_resolution(buckets: List[Tuple[int, int]], size: Tuple[int, int]) -> Tuple[int, int]:
    img_ar = size[0] / size[1]

    def distance(reso: Tuple[int, int]) -> float:
        return abs(img_ar - reso[0]/reso[1])

    return min(buckets, key=distance)


def get_bucket_reso(
    image_size: Tuple[int, int],
    max_resolution: Optional[Union[Tuple[int, int], int]] = 1024,
    max_aspect_ratio: Optional[float] = 1.1,
    divisible: Optional[int] = 32,
    buckets: Optional[List[Tuple[int, int]]] = None,
):
    r"""
    Get the closest resolution to the image's resolution from the buckets. If the image's aspect ratio is too
    different from the closest resolution, then return the around resolution based on the max resolution.
    :param image: The image to be resized.
    :param buckets: The buckets of resolutions to choose from. Default to SDXL_BUCKETS. Set None to use max_resolution.
    :param max_resolution: The max resolution to be used to calculate the around resolution. It's used to calculate the 
        around resolution when `buckets` is None or no bucket can contain that image without exceeding the max aspect ratio.
        Default to 1024. Set `-1` to auto calculate from the buckets' max resolution. Set None to disable.
        Set None to auto calculate from the buckets' max resolution.
    :param max_aspect_ratio: The max aspect ratio difference between the image and the closest resolution. Default to 1.1.
        Set None to disable.
    :param divisible: The divisible number of bucket resolutions. Default to 32.
    :return: The closest resolution to the image's resolution.
    """
    if not buckets and (not max_resolution or max_resolution == -1):
        raise ValueError(
            "Either `buckets` or `max_resolution` must be provided.")

    img_w, img_h = image_size
    clo_reso = closest_resolution(buckets, image_size) if buckets else around_reso(
        img_w, img_h, reso=max_resolution, divisible=divisible)
    max_resolution = max(buckets, key=lambda x: x[0]*x[1]) if buckets and max_resolution == -1 else max_resolution

    # Handle special resolutions
    if img_w < clo_reso[0] or img_h < clo_reso[1]:
        new_w = img_w // divisible * divisible
        new_h = img_h // divisible * divisible
        clo_reso = (new_w, new_h)
    elif max_aspect_ratio and aspect_ratio_diff((img_w, img_h), clo_reso) >= max_aspect_ratio:
        if buckets and max_resolution:
            clo_reso = around_reso(
                img_w, img_h, reso=max_resolution, divisible=divisible)
        else:
            log_utils.warn(
                f"An image has aspect ratio {img_w/img_h:.2f} which is too different from the closest resolution {clo_reso[0]/clo_reso[1]}. You may lower the `divisible` to avoid this.")

    return clo_reso
