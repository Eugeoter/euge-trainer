import io
import struct
import re
import cv2
import random
import os
import numpy as np
from ml_collections import ConfigDict
from PIL import Image, ExifTags
from typing import Tuple, Union, Literal
from torchvision import transforms

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

RESAMPLING_METHODS = {
    "nearest": (Image.NEAREST, cv2.INTER_NEAREST),
    "box": (Image.BOX, cv2.INTER_AREA),
    "bilinear": (Image.BILINEAR, cv2.INTER_LINEAR),
    "hamming": (Image.HAMMING, cv2.INTER_LINEAR),
    "bicubic": (Image.BICUBIC, cv2.INTER_CUBIC),
    "lanczos": (Image.LANCZOS, cv2.INTER_LANCZOS4),
}


def cfg(**kwargs):
    return ConfigDict(initial_dictionary=kwargs)


def get_platform() -> Literal['Windows', 'Linux', 'macOS', 'Other POSIX', 'Unknown']:
    if os.name == 'nt':
        return "Windows"
    elif os.name == 'posix':
        import platform
        if platform.system() == 'Linux':
            return "Linux"
        elif platform.system() == 'Darwin':
            return "macOS"
        else:
            return "Other POSIX"
    else:
        return "Unknown"


def load_image(image_path, type: Literal['pil', 'numpy'] = 'numpy', mode: str = 'RGB'):
    image = Image.open(image_path)
    if mode is not None and not image.mode == mode:
        image = image.convert(mode)
    if type == 'numpy':
        image = np.array(image, np.uint8)  # (H, W, C)
    return image


def rotate_image_straight(image: Image) -> Image:
    r"""
    Rotate the image to correct the orientation.
    """
    from PIL import ExifTags
    exif: Image.Exif = image.getexif()
    if exif:
        orientation_tag = {v: k for k, v in ExifTags.TAGS.items()}['Orientation']
        orientation = exif.get(orientation_tag)
        degree = {
            3: 180,
            6: 270,
            8: 90,
        }.get(orientation)
        if degree:
            image = image.rotate(degree, expand=True)
    return image


def fill_transparency(image: Union[Image.Image, np.ndarray], bg_color: Tuple[int, int, int] = (255, 255, 255)):
    r"""
    Fill the transparent part of an image with a background color.
    Please pay attention that this function doesn't change the image type.
    """
    if isinstance(image, Image.Image):
        # Only process if image has transparency
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
            alpha = image.convert('RGBA').split()[-1]

            # Create a new background image of our matt color.
            # Must be RGBA because paste requires both images have the same format
            # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
            bg = Image.new("RGBA", image.size, bg_color + (255,))
            bg.paste(image, mask=alpha)
            return bg
        else:
            return image
    elif isinstance(image, np.ndarray):
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            bg = np.full_like(image, bg_color + (255,))
            bg[:, :, :3] = image[:, :, :3]
            return bg
        else:
            return image


def convert_to_rgb(image: Union[Image.Image, np.ndarray], bg_color=(255, 255, 255)):
    r"""
    Convert an image to RGB mode and fix transparency conversion if needed.
    """
    image = fill_transparency(image, bg_color)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    elif isinstance(image, np.ndarray):
        return image[:, :, :3]


def color_aug(image: np.ndarray) -> np.ndarray:
    # self.color_aug_method = albu.OneOf(
    #     [
    #         albu.HueSaturationValue(8, 0, 0, p=0.5),
    #         albu.RandomGamma((95, 105), p=0.5),
    #     ],
    #     p=0.33,
    # )
    hue_shift_limit = 8

    # remove dependency to albumentations
    if random.random() <= 0.33:
        if random.random() > 0.5:
            # hue shift
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue_shift = random.uniform(-hue_shift_limit, hue_shift_limit)
            if hue_shift < 0:
                hue_shift = 180 + hue_shift
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
            image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        else:
            # random gamma
            gamma = random.uniform(0.95, 1.05)
            image = np.clip(image**gamma, 0, 255).astype(np.uint8)

    return image


def resize_if_needed(image: Union[Image.Image, np.ndarray], target_size: Tuple[int, int], resampling='lanczos'):
    if isinstance(image, Image.Image):
        if image.width != target_size[0] or image.height != target_size[1]:
            image = image.resize(target_size, resample=get_resampling(resampling, 'pil'))
    elif isinstance(image, np.ndarray):  # (H, W, C)
        if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
            image = cv2.resize(image, target_size, interpolation=get_resampling(resampling, 'numpy'))
    return image


def get_resampling(resampling: str, type: Literal['pil', 'numpy'] = 'numpy'):
    if resampling not in RESAMPLING_METHODS:
        raise ValueError(f"Unknown resampling method: {resampling}, available methods: {', '.join(RESAMPLING_METHODS.keys())}")
    return RESAMPLING_METHODS[resampling][0 if type == 'pil' else 1]


def crop_ltrb_if_needed(image: Union[Image.Image, np.ndarray], crop_ltrb):
    wid, hei = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
    # if crop_ltrb exceeds the image size, raise an error
    if crop_ltrb[0] < 0 or crop_ltrb[1] < 0 or crop_ltrb[2] > wid or crop_ltrb[3] > hei:
        raise ValueError(f"crop_ltrb {crop_ltrb} exceeds the image size {wid}x{hei}")
    if crop_ltrb != (0, 0, wid, hei):
        if isinstance(image, Image.Image):
            image = image.crop(crop_ltrb)
        else:
            image = image[crop_ltrb[1]:crop_ltrb[3], crop_ltrb[0]:crop_ltrb[2]]
    return image


def crop_if_needed(image: Union[Image.Image, np.ndarray], target_size, max_ar=None):
    image_size = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
    img_w, img_h = image_size
    tar_w, tar_h = target_size

    ar_image = img_w / img_h
    ar_target = tar_w / tar_h

    if max_ar is not None and aspect_ratio_diff(image_size, target_size) > max_ar:
        if ar_image < ar_target:
            new_height = img_w / ar_target * max_ar
            new_width = img_w
        else:
            new_width = img_h * ar_target / max_ar
            new_height = img_h

        left = max(0, int((img_w - new_width) / 2))
        top = max(0, int((img_h - new_height) / 2))
        right = int(left + new_width)
        bottom = int(top + new_height)
        crop_ltrb = (left, top, right, bottom)

        if isinstance(image, Image.Image):
            image = image.crop(crop_ltrb)
        else:
            image = image[top:bottom, left:right]
    else:
        crop_ltrb = (0, 0, img_w, img_h)

    return image, crop_ltrb


def center_crop_if_needed(image: Union[Image.Image, np.ndarray], target_size):
    image_size = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
    img_w, img_h = image_size
    tar_w, tar_h = target_size

    ar_image = img_w / img_h
    ar_target = tar_w / tar_h

    if ar_image < ar_target:
        new_height = img_w / ar_target
        new_width = img_w
    else:
        new_width = img_h * ar_target
        new_height = img_h

    left = max(0, int((img_w - new_width) / 2))
    top = max(0, int((img_h - new_height) / 2))
    right = int(left + new_width)
    bottom = int(top + new_height)
    crop_ltrb = (left, top, right, bottom)

    if isinstance(image, Image.Image):
        image = image.crop(crop_ltrb)
    else:
        image = image[top:bottom, left:right]

    return image, crop_ltrb


def convert_size_if_needed(size: Union[str, Tuple[int, int]]):
    if isinstance(size, str):
        size = tuple(map(int, size.split("x")))
    return size


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


def _convertToPx(value):
    matched = re.match(r"(\d+(?:\.\d+)?)?([a-z]*)$", value)
    if not matched:
        raise ValueError("unknown length value: %s" % value)

    length, unit = matched.groups()
    if unit == "":
        return float(length)
    elif unit == "cm":
        return float(length) * 96 / 2.54
    elif unit == "mm":
        return float(length) * 96 / 2.54 / 10
    elif unit == "in":
        return float(length) * 96
    elif unit == "pc":
        return float(length) * 96 / 6
    elif unit == "pt":
        return float(length) * 96 / 6
    elif unit == "px":
        return float(length)

    raise ValueError("unknown unit type: %s" % unit)


def get_image_size(img_path):
    """
    Return (width, height) for a given img file content
    no requirements
    :type filepath: Union[bytes, str, pathlib.Path]
    :rtype Tuple[int, int]
    """
    height = -1
    width = -1

    if isinstance(img_path, io.BytesIO):  # file-like object
        fhandle = img_path
    else:
        fhandle = open(img_path, 'rb')

    try:
        head = fhandle.read(31)
        size = len(head)
        # handle GIFs
        if size >= 10 and head[:6] in (b'GIF87a', b'GIF89a'):
            # Check to see if content_type is correct
            try:
                width, height = struct.unpack("<hh", head[6:10])
            except struct.error:
                raise ValueError("Invalid GIF file")
        # see png edition spec bytes are below chunk length then and finally the
        elif size >= 24 and head.startswith(b'\211PNG\r\n\032\n') and head[12:16] == b'IHDR':
            try:
                width, height = struct.unpack(">LL", head[16:24])
            except struct.error:
                raise ValueError("Invalid PNG file")
        # Maybe this is for an older PNG version.
        elif size >= 16 and head.startswith(b'\211PNG\r\n\032\n'):
            # Check to see if we have the right content type
            try:
                width, height = struct.unpack(">LL", head[8:16])
            except struct.error:
                raise ValueError("Invalid PNG file")
        # handle JPEGs
        elif size >= 2 and head.startswith(b'\377\330'):
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf or ftype in [0xc4, 0xc8, 0xcc]:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except (struct.error, TypeError):
                raise ValueError("Invalid JPEG file")
        # handle JPEG2000s
        elif size >= 12 and head.startswith(b'\x00\x00\x00\x0cjP  \r\n\x87\n'):
            fhandle.seek(48)
            try:
                height, width = struct.unpack('>LL', fhandle.read(8))
            except struct.error:
                raise ValueError("Invalid JPEG2000 file")
        # handle big endian TIFF
        elif size >= 8 and head.startswith(b"\x4d\x4d\x00\x2a"):
            offset = struct.unpack('>L', head[4:8])[0]
            fhandle.seek(offset)
            ifdsize = struct.unpack(">H", fhandle.read(2))[0]
            for i in range(ifdsize):
                tag, datatype, count, data = struct.unpack(">HHLL", fhandle.read(12))
                if tag == 256:
                    if datatype == 3:
                        width = int(data / 65536)
                    elif datatype == 4:
                        width = data
                    else:
                        raise ValueError("Invalid TIFF file: width column data type should be SHORT/LONG.")
                elif tag == 257:
                    if datatype == 3:
                        height = int(data / 65536)
                    elif datatype == 4:
                        height = data
                    else:
                        raise ValueError("Invalid TIFF file: height column data type should be SHORT/LONG.")
                if width != -1 and height != -1:
                    break
            if width == -1 or height == -1:
                raise ValueError("Invalid TIFF file: width and/or height IDS entries are missing.")
        elif size >= 8 and head.startswith(b"\x49\x49\x2a\x00"):
            offset = struct.unpack('<L', head[4:8])[0]
            fhandle.seek(offset)
            ifdsize = struct.unpack("<H", fhandle.read(2))[0]
            for i in range(ifdsize):
                tag, datatype, count, data = struct.unpack("<HHLL", fhandle.read(12))
                if tag == 256:
                    width = data
                elif tag == 257:
                    height = data
                if width != -1 and height != -1:
                    break
            if width == -1 or height == -1:
                raise ValueError("Invalid TIFF file: width and/or height IDS entries are missing.")
        # handle little endian BigTiff
        elif size >= 8 and head.startswith(b"\x49\x49\x2b\x00"):
            bytesize_offset = struct.unpack('<L', head[4:8])[0]
            if bytesize_offset != 8:
                raise ValueError('Invalid BigTIFF file: Expected offset to be 8, found {} instead.'.format(offset))
            offset = struct.unpack('<Q', head[8:16])[0]
            fhandle.seek(offset)
            ifdsize = struct.unpack("<Q", fhandle.read(8))[0]
            for i in range(ifdsize):
                tag, datatype, count, data = struct.unpack("<HHQQ", fhandle.read(20))
                if tag == 256:
                    width = data
                elif tag == 257:
                    height = data
                if width != -1 and height != -1:
                    break
            if width == -1 or height == -1:
                raise ValueError("Invalid BigTIFF file: width and/or height IDS entries are missing.")

        # handle SVGs
        elif size >= 5 and (head.startswith(b'<?xml') or head.startswith(b'<svg')):
            fhandle.seek(0)
            data = fhandle.read(1024)
            try:
                data = data.decode('utf-8')
                width = re.search(r'[^-]width="(.*?)"', data).group(1)
                height = re.search(r'[^-]height="(.*?)"', data).group(1)
            except Exception:
                raise ValueError("Invalid SVG file")
            width = _convertToPx(width)
            height = _convertToPx(height)

        # handle Netpbm
        elif head[:1] == b"P" and head[1:2] in b"123456":
            fhandle.seek(2)
            sizes = []

            while True:
                next_chr = fhandle.read(1)

                if next_chr.isspace():
                    continue

                if next_chr == b"":
                    raise ValueError("Invalid Netpbm file")

                if next_chr == b"#":
                    fhandle.readline()
                    continue

                if not next_chr.isdigit():
                    raise ValueError("Invalid character found on Netpbm file")

                size = next_chr
                next_chr = fhandle.read(1)

                while next_chr.isdigit():
                    size += next_chr
                    next_chr = fhandle.read(1)

                sizes.append(int(size))

                if len(sizes) == 2:
                    break

                fhandle.seek(-1, os.SEEK_CUR)
            width, height = sizes
        elif head.startswith(b"RIFF") and head[8:12] == b"WEBP":
            if head[12:16] == b"VP8 ":
                width, height = struct.unpack("<HH", head[26:30])
            elif head[12:16] == b"VP8X":
                width = struct.unpack("<I", head[24:27] + b"\0")[0]
                height = struct.unpack("<I", head[27:30] + b"\0")[0]
            elif head[12:16] == b"VP8L":
                b = head[21:25]
                width = (((b[1] & 63) << 8) | b[0]) + 1
                height = (((b[3] & 15) << 10) | (b[2] << 2) | ((b[1] & 192) >> 6)) + 1
            else:
                raise ValueError("Unsupported WebP file")

    finally:
        fhandle.close()

    return width, height
