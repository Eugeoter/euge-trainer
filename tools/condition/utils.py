import os
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_project_root(root):
    global PROJECT_ROOT
    PROJECT_ROOT = Path(root)
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    return PROJECT_ROOT


def set_device(device):
    global DEVICE
    DEVICE = torch.device(device)
    return DEVICE
