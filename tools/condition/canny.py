import cv2
import numpy as np


def get_canny(np_img: np.ndarray, threshold1=None, threshold2=None) -> np.ndarray:
    if not isinstance(np_img, np.ndarray):
        raise ValueError(f"Expected image to be a numpy array, got {type(np_img)}")

    if len(np_img.shape) == 2:
        gray = np_img
    elif len(np_img.shape) == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    elif len(np_img.shape) == 4:
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Invalid image shape: {np_img.shape}")

    if threshold1 is None and threshold2 is None:  # auto threshold
        v = np.median(gray)
        sigma = 0.33
        threshold1 = int(max(0, (1.0 - sigma) * v))
        threshold2 = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges
