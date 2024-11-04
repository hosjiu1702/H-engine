import argparse
import random
import string
import os
from pathlib import Path
import numpy as np
import torch
import accelerate


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