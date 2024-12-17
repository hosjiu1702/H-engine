from typing import Union, List, Tuple
import argparse
import random
import string
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import PIL
from PIL import ImageOps
import torch
import accelerate
import torchvision
from torchvision.transforms.functional import pil_to_tensor
from einops import rearrange
from matplotlib import pyplot as plt
from diffusers import AutoencoderKL

from src.models.attention_processor import (
    SkipAttnProcessor,
    AttnProcessor2_0 as AttnProcessor
)
from src.models.emasc import EMASC


def resize_keep_ratio(img: PIL.Image.Image, size: Tuple = (384, 512)) -> PIL.Image.Image:
    output = ImageOps.fit(img, size)
    return output


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


def show(img: Union[torch.Tensor, PIL.Image.Image], print_shape: bool = False, title: str = ""):
    if isinstance(img, PIL.Image.Image):
        img = pil_to_tensor(img)

    if img.ndim == 3:
        img = rearrange(img, 'c h w -> h w c')
    else:
        raise ValueError('Only support for RGB image for now or Tensor 3D.')

    if print_shape:
        print(img.shape)

    plt.title(title)
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


def mask_features(features: list, mask: torch.Tensor):
    """
    Mask features with the given mask. (EMASC from LaDI-VTON paper)
    """

    for i, feature in enumerate(features):
        # Resize the mask to the feature size.
        mask = torch.nn.functional.interpolate(mask, size=feature.shape[-2:])

        # Mask the feature.
        features[i] = feature * (1 - mask)

    return features


@torch.inference_mode()
def extract_save_vae_images(vae: AutoencoderKL, emasc: EMASC, test_dataloader: torch.utils.data.DataLoader,
                            int_layers: List[int], output_dir: str, order: str, save_name: str,
                            emasc_type: str) -> None:
    """
    Extract and save image using only VAE or VAE + EMASC
    """
    # Create output directory
    save_path = os.path.join(output_dir, f"{save_name}_{order}")
    os.makedirs(save_path, exist_ok=True)

    for idx, batch in enumerate(tqdm(test_dataloader)):
        category = batch["category"]

        if emasc_type != "none":
            # Extract intermediate features from 'im_mask' and encode image
            posterior_im, _ = vae.encode(batch["image"])
            _, intermediate_features = vae.encode(batch["im_mask"])
            # intermediate_features = [intermediate_features[i] for i in int_layers]

            # Use EMASC
            processed_intermediate_features = emasc(intermediate_features)

            processed_intermediate_features = mask_features(processed_intermediate_features, batch["inpaint_mask"])
            latents = posterior_im.latent_dist.sample()
            generated_images = vae.decode(
                latents,
                processed_intermediate_features,
                # int_layers
            ).sample
        else:
            # Encode and decode image without EMASC
            posterior_im = vae.encode(batch["image"])
            latents = posterior_im.latent_dist.sample()
            generated_images = vae.decode(latents).sample

        # Save images
        for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
            gen_image = (gen_image + 1) / 2  # [-1, 1] -> [0, 1]
            if not os.path.exists(os.path.join(save_path, cat)):
                os.makedirs(os.path.join(save_path, cat))
            torchvision.utils.save_image(gen_image, os.path.join(save_path, cat, name))


def is_image(filename: str) -> bool:
    VALID_IMAGE_EXTENSION = {'.png', '.jpg'}
    _, file_ext = os.path.splitext(filename)
    if file_ext in VALID_IMAGE_EXTENSION:
        return True
    return False