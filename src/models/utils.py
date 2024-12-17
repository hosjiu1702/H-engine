from typing import Tuple
import os
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler
from src.models.autoencoder_kl import AutoencoderKLForEmasc
from src.models.attention_processor import SkipAttnProcessor
from src.utils import get_project_root, init_attn_processor
from src.preprocess import apply_net
from src.dataset.vitonhd import VITONHDDataset


PROJECT_ROOT_PATH = get_project_root()


def download_model(repo_id, ckpt_name, model_name):
    # UNET
    unet_path = hf_hub_download(
        repo_id=repo_id,
        subfolder=os.path.join(ckpt_name, 'unet'),
        filename='diffusion_pytorch_model.safetensors',
        local_dir=os.path.join(PROJECT_ROOT_PATH, 'checkpoints', model_name)
    )
    hf_hub_download(
        repo_id=repo_id,
        subfolder=os.path.join(ckpt_name, 'unet'),
        filename='config.json',
        local_dir=os.path.join(PROJECT_ROOT_PATH, 'checkpoints', model_name)
    )

    # VAE
    hf_hub_download(
        repo_id=repo_id,
        subfolder=os.path.join(ckpt_name, 'vae'),
        filename='diffusion_pytorch_model.safetensors',
        local_dir=os.path.join(PROJECT_ROOT_PATH, 'checkpoints', model_name)
    )
    hf_hub_download(
        repo_id=repo_id,
        subfolder=os.path.join(ckpt_name, 'vae'),
        filename='config.json',
        local_dir=os.path.join(PROJECT_ROOT_PATH, 'checkpoints', model_name)
    )

    # SCHEDULER
    hf_hub_download(
        repo_id=repo_id,
        subfolder=os.path.join(ckpt_name, 'scheduler'),
        filename='scheduler_config.json',
        local_dir=os.path.join(PROJECT_ROOT_PATH, 'checkpoints', model_name)
    )

    # model_index.json
    hf_hub_download(
        repo_id=repo_id,
        subfolder=ckpt_name,
        filename='model_index.json',
        local_dir=os.path.join(PROJECT_ROOT_PATH, 'checkpoints', model_name)
    )

    model_path = os.path.dirname(os.path.dirname(unet_path))
    
    return model_path


def load_model(model_path: str, dtype=torch.float16):
    vae = AutoencoderKLForEmasc.from_pretrained(
        model_path,
        subfolder='vae',
        torch_dtype=dtype
    )

    scheduler = DDPMScheduler.from_pretrained(
        model_path,
        subfolder='scheduler'
    )

    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder='unet',
        torch_dtype=dtype
    )

    init_attn_processor(unet, cross_attn_cls=SkipAttnProcessor)

    return unet, vae, scheduler


def get_densepose_map(img_path: str, size: Tuple = (384, 512)) -> Image.Image:
    img = Image.open(img_path)
    img = ImageOps.fit(img, size=size)

    args = apply_net.create_argument_parser().parse_args((
        'show',
        os.path.join(PROJECT_ROOT_PATH, 'configs/densepose_rcnn_R_50_FPN_s1x.yaml'),
        os.path.join(PROJECT_ROOT_PATH, 'checkpoints/densepose/model_final_162be9.pkl'),
        img_path,
        'dp_segm',
        '-v'
    ))
    densepose_np = args.func(args, img)[0]

    return Image.fromarray(densepose_np[:, :, ::-1])


def preprocess_image(img: Image.Image, w: int, h: int) -> torch.Tensor:
    img = VITONHDDataset.preprocess(img, w, h)
    return img