import cv2
import numpy as np
import os
from pathlib import Path
from waifuset import logging
from .utils import MODELS_DIR

CONTROL_TYPE = "hed"

PROTOTXT_NAME = "deploy.prototxt"
CAFFEMODEL_NAME = "hed_pretrained_bsds.caffemodel"

HED_NETWORK = None

LOGGER = logging.getLogger(CONTROL_TYPE)

# LOGGER.debug(cv2.getBuildInformation())


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def get_hed(np_img: np.ndarray, prototxt=PROTOTXT_NAME, caffemodel=CAFFEMODEL_NAME) -> np.ndarray:
    r"""
    Get the HED (Holistically-Nested Edge Detection) of an numpy image.
    """

    global HED_NETWORK
    if HED_NETWORK is None:
        # Load the model.
        cv2.dnn_registerLayer('Crop', CropLayer)
        prototxt_path = os.path.join(MODELS_DIR, CONTROL_TYPE, prototxt)
        caffemodel_path = os.path.join(MODELS_DIR, CONTROL_TYPE, caffemodel)
        HED_NETWORK = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

        HED_NETWORK.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        HED_NETWORK.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    height, width = np_img.shape[:2]

    inp = cv2.dnn.blobFromImage(np_img, scalefactor=1.0, size=(width, height),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    HED_NETWORK.setInput(inp)
    out = HED_NETWORK.forward()

    out = out[0, 0]
    out = cv2.resize(out, (np_img.shape[1], np_img.shape[0]))

    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)

    # logging.debug("After target:", HED_NETWORK.getPerfProfile()[0])

    return out


if __name__ == "__main__":
    img_path = r"d:\AI\datasets\aid\images\dataset\alchemy stars\alchemy_stars_8.webp"

    image = cv2.imread(img_path)
    hed = get_hed(image)

    cv2.imshow("HED", hed)
    cv2.waitKey(0)
