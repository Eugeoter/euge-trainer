import os
import torch
import numpy as np
import torch.nn as nn
from typing import Literal
from huggingface_hub import hf_hub_download
from einops import rearrange
from waifuset import logging
from .utils import MODELS_DIR, DEVICE

CONTROL_TYPE = "lineart"
LOGGER = logging.getLogger(CONTROL_TYPE)

DEFAULT_MODEL_NAME = "sk_model.pth"
COARSE_MODEL_NAME = "sk_model2.pth"

MODEL_REPO_ID = "lllyasviel/Annotators"

DEFAULT_DETECTOR = None
COARSE_DETECTOR = None

norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features)
                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, 64, 7),
                  norm_layer(64),
                  nn.ReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                       norm_layer(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                       norm_layer(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartDetector:
    def __init__(self, model_name, device=DEVICE):
        self.model = None
        self.model_name = model_name
        self.device = device

    def load_model(self, name):
        model_dir = os.path.join(MODELS_DIR, CONTROL_TYPE)
        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_path):
            LOGGER.info(f"Downloading model {name}...")
            model_path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=name,
                local_dir=model_dir
            )
        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        self.model = model.to(self.device)

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image):
        if self.model is None:
            self.load_model(self.model_name)
        self.model.to(self.device)

        assert input_image.ndim == 3
        image = input_image
        with torch.no_grad():
            image = torch.from_numpy(image).float().to(self.device)
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            line = self.model(image)[0][0]

            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

            return line


def get_lineart(np_img: np.ndarray, model_name: Literal["sk_model.pth", "sk_model2.pth"] = DEFAULT_MODEL_NAME, white_bg=False, device=DEVICE) -> np.ndarray:
    r"""
    Get the lineart of an numpy image.
    """
    global DEFAULT_DETECTOR, COARSE_DETECTOR
    if model_name == DEFAULT_MODEL_NAME:
        if DEFAULT_DETECTOR is None:
            DEFAULT_DETECTOR = LineartDetector(model_name, device)
        line = DEFAULT_DETECTOR(np_img)
    elif model_name == COARSE_MODEL_NAME:
        if COARSE_DETECTOR is None:
            COARSE_DETECTOR = LineartDetector(model_name, device)
        line = COARSE_DETECTOR(np_img)
    else:
        raise ValueError(f"Model name {model_name} not supported. Please choose from {DEFAULT_MODEL_NAME} or {COARSE_MODEL_NAME}.")
    line = DEFAULT_DETECTOR(np_img)
    if not white_bg:
        line = 255 - line
    return line
