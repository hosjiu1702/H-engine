from typing import Union
import argparse
import random
import string
import os
from pathlib import Path
import numpy as np
import PIL
import torch
import accelerate
from torchvision.transforms.functional import pil_to_tensor
from einops import rearrange
from matplotlib import pyplot as plt

from src.models.attention_processor import (
    SkipAttnProcessor,
    AttnProcessor2_0 as AttnProcessor
)


# Copied from https://github.com/miccunifi/ladi-vton/blob/master/src/utils/set_seeds.py
def set_seed(seed: int):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    accelerate.utils.set_seed(seed)


def show(img: Union[torch.Tensor, PIL.Image.Image], print_shape: bool = False):
    if isinstance(img, PIL.Image.Image):
        img = pil_to_tensor(img)

    if img.ndim == 3:
        img = rearrange(img, 'c h w -> h w c')
    else:
        raise ValueError('Only support for RGB image for now or Tensor 3D.')

    if print_shape:
        print(img.shape)

    plt.imshow(img)


def set_train(module: torch.nn.Module, is_train: bool = True):
    if is_train:
        module.requires_grad_(True)
    else:
        module.requires_grad_(False)


def use_gradient_accumulation(val: int) -> bool:
    return True if val > 1 else False


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def total_trainable_params(model: torch.nn.Module):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


# https://stackoverflow.com/a/2257449/7890329
def generate_rand_chars(size=10):
    return ''.join([random.choice(string.ascii_lowercase + string.digits) for _ in range(size)])


# https://stackoverflow.com/a/43357954/7890329
def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        argparse.ArgumentTypeError('boolean value expected.')


def init_attn_processor(
        unet: torch.nn.Module, 
        cross_attn_cls=SkipAttnProcessor,
    ):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = cross_attn_cls()
                                                    
    unet.set_attn_processor(attn_procs)