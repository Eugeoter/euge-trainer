import cv2
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Literal
from scipy import ndimage
from waifuset import logging
from keras.models import load_model
from .utils import MODELS_DIR

CONTROL_TYPE = "sketch_keras"
LOGGER = logging.get_logger(CONTROL_TYPE)
DEFAULT_MODEL_NAME = "mod.h5"
MODEL = None


def get_normal_map(img):
    img = img.astype(np.float32)
    img = img / 255.0
    img = - img + 1
    img[img < 0] = 0
    img[img > 1] = 1
    return img


def get_gray_map(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    highPass = gray.astype(np.float32)
    highPass = highPass / 255.0
    highPass = 1 - highPass
    highPass = highPass[None]
    return highPass.transpose((1, 2, 0))


def get_light_map(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    highPass = gray.astype(int) - blur.astype(int)
    highPass = highPass.astype(np.float32)
    highPass = highPass / 128.0
    highPass = highPass[None]
    return highPass.transpose((1, 2, 0))


def get_light_map_single(img):
    gray = img
    gray = gray[None]
    gray = gray.transpose((1, 2, 0))
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = gray.reshape((gray.shape[0], gray.shape[1]))
    highPass = gray.astype(int) - blur.astype(int)
    highPass = highPass.astype(np.float32)
    highPass = highPass / 128.0
    return highPass


def get_light_map_drawer(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    highPass = gray.astype(int) - blur.astype(int) + 255
    highPass[highPass < 0] = 0
    highPass[highPass > 255] = 255
    highPass = highPass.astype(np.float32)
    highPass = highPass / 255.0
    highPass = 1 - highPass
    highPass = highPass[None]
    return highPass.transpose((1, 2, 0))


def get_light_map_drawer2(img):
    ret = img.copy()
    ret = ret.astype(np.float32)
    ret[:, :, 0] = get_light_map_drawer3(img[:, :, 0])
    ret[:, :, 1] = get_light_map_drawer3(img[:, :, 1])
    ret[:, :, 2] = get_light_map_drawer3(img[:, :, 2])
    ret = np.amax(ret, 2)
    return ret


def get_light_map_drawer3(img):
    gray = img
    blur = cv2.blur(gray, ksize=(5, 5))
    highPass = gray.astype(int) - blur.astype(int) + 255
    highPass[highPass < 0] = 0
    highPass[highPass > 255] = 255
    highPass = highPass.astype(np.float32)
    highPass = highPass / 255.0
    highPass = 1 - highPass
    return highPass


def normalize_pic(img):
    img = img / np.max(img)
    return img


def superlize_pic(img):
    img = img * 2.33333
    img[img > 1] = 1
    return img


def mask_pic(img, mask):
    mask_mat = mask
    mask_mat = mask_mat.astype(np.float32)
    mask_mat = cv2.GaussianBlur(mask_mat, (0, 0), 1)
    mask_mat = mask_mat / np.max(mask_mat)
    mask_mat = mask_mat * 255
    mask_mat[mask_mat < 255] = 0
    mask_mat = mask_mat.astype(np.uint8)
    mask_mat = cv2.GaussianBlur(mask_mat, (0, 0), 3)
    mask_mat = get_gray_map(mask_mat)
    mask_mat = normalize_pic(mask_mat)
    mask_mat = resize_img_512(mask_mat)
    super_from = np.multiply(img, mask_mat)
    return super_from


def resize_img_512(img):
    zeros = np.zeros((512, 512, img.shape[2]), dtype=np.float32)
    zeros[:img.shape[0], :img.shape[1]] = img
    return zeros


def resize_img_512_3d(img):
    zeros = np.zeros((1, 3, 512, 512), dtype=np.float32)
    zeros[0, 0: img.shape[0], 0: img.shape[1], 0: img.shape[2]] = img
    return zeros.transpose((1, 2, 3, 0))


def get_active_img(img):
    mat = img.astype(np.float32)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    return mat


def get_active_img_fil(img):
    mat = img.astype(np.float32)
    mat[mat < 0.18] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    return mat


def show_active_img_and_save(name, img, path):
    mat = img.astype(np.float32)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    cv2.imshow(name, mat)
    cv2.imwrite(path, mat)
    return


def denoise_mat(img, i):
    return ndimage.median_filter(img, i)


def get_active_img_denoise(img):
    mat = img.astype(np.float32)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    return mat


def show_active_img_and_save_denoise(name, img, path):
    mat = img.astype(np.float32)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    cv2.imshow(name, mat)
    cv2.imwrite(path, mat)
    return


def get_active_img_denoise_filter(img):
    mat = img.astype(np.float32)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    return mat


def show_active_img_and_save_denoise_filter(name, img, path):
    mat = img.astype(np.float32)
    mat[mat < 0.18] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    cv2.imshow(name, mat)
    cv2.imwrite(path, mat)
    return


def get_active_img_denoise_filter2(img):
    mat = img.astype(np.float32)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    return mat


def show_active_img_and_save_denoise_filter2(name, img, path):
    mat = img.astype(np.float32)
    mat[mat < 0.1] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    cv2.imshow(name, mat)
    cv2.imwrite(path, mat)
    return


def show_active_img(name, img):
    mat = img.astype(np.float32)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    cv2.imshow(name, mat)
    return


def get_active_img(img):
    mat = img.astype(np.float32)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    return mat


def get_active_img_fil(img):
    mat = img.astype(np.float32)
    mat[mat < 0.18] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    return mat


def show_double_active_img(name, img):
    mat = img.astype(np.float32)
    mat = mat * 128.0
    mat = mat + 127.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    cv2.imshow(name, mat.astype(np.uint8))
    return


def debug_pic_helper():
    for index in range(1130):
        gray_path = 'data\\gray\\'+str(index)+'.jpg'
        color_path = 'data\\color\\' + str(index) + '.jpg'

        mat_color = cv2.imread(color_path)
        mat_color = get_light_map(mat_color)
        mat_color = normalize_pic(mat_color)
        mat_color = resize_img_512(mat_color)
        show_double_active_img('mat_color', mat_color)

        mat_gray = cv2.imread(gray_path)
        mat_gray = get_gray_map(mat_gray)
        mat_gray = normalize_pic(mat_gray)
        mat_gray = resize_img_512(mat_gray)
        show_active_img('mat_gray', mat_gray)

        cv2.waitKey(1000)


def get_sketch(np_img, img_type: Literal['colored', 'enhenced', 'pured', 'normal'] = 'colored') -> np.ndarray:
    global MODEL
    if MODEL is None:
        model_path = os.path.join(MODELS_DIR, CONTROL_TYPE, DEFAULT_MODEL_NAME)
        LOGGER.info(f"Loading sketch model from {logging.yellow(model_path)}")
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            MODEL = load_model(model_path)

    width, height = float(np_img.shape[1]), float(np_img.shape[0])
    if width > height:
        np_img = cv2.resize(np_img, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
        new_width, new_height = 512, int(512 / width * height)
    else:
        np_img = cv2.resize(np_img, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
        new_width, new_height = int(512 / height * width), 512

    np_img = np_img.transpose((2, 0, 1))
    light_map = np.zeros(np_img.shape, dtype=np.float32)

    for channel in range(3):
        light_map[channel] = get_light_map_single(np_img[channel])

    light_map = normalize_pic(light_map)
    light_map = resize_img_512_3d(light_map)

    # Predict on GPU
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        line_mat = MODEL.predict(light_map, batch_size=1)

    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:int(new_height), 0:int(new_width), :]

    if img_type == 'colored':
        return get_active_img(line_mat)
    line_mat = np.amax(line_mat, 2)

    if img_type == 'enhenced':
        return get_active_img_denoise_filter2(line_mat)
    if img_type == 'pured':
        return get_active_img_denoise_filter(line_mat)
    if img_type == 'normal':
        return get_active_img_denoise(line_mat)

    raise ValueError(f"Unknown image type: {img_type}")


if __name__ == '__main__':
    get_sketch(cv2.imread('data\\color\\0.jpg'))
