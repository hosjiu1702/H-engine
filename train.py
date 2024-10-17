# Modified from:
# https://github.com/miccunifi/ladi-vton/blob/master/src/train_vto.py
# https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py
# https://github.com/yisol/IDM-VTON/blob/1b39608bf3b6f075b21562e86302dcefd6989fc5/train_xl.py

import itertools
import torch
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from src.utils.utils import set_seed, set_train
from src.models.ip_adapter.attention_processor import (
    AttnProcessor2_0 as AttnProcessor,
    IPAttnProcessor2_0 as IPAttnProcessor
)
from src.models.ip_adaper.resampler import Resampler


def main():
    args = None # update later
    
    # Init Acclerator object for tracking training process
    project_config = ProjectConfiguration(args.project_dir, args.logging_dir)
    accelerator = Accelerator(
                log_with=args.report_to,
                project_config=project_config
            )

    # For reproducibility
    if args.set_seed:
        set_seed(args.seed)

    # Load diffusion-related components
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae', torch_dtype=torch.float16) # float16 vs float32 -> which one to choose?
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet')

    # Load IP-Adapter pretrained weights
    state_dict = torch.load(args.pretrained_ip_adapter_path, map_location='cpu')

    # Load IP-Adapter for joint training with Denoising U-net.
    # We load the existing adapter modules from the original U-net
    # and changes its cross-attention processors from IP-Adapter
    attn_procs = dict()
    unet_state_dict = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith('attn1.processor') else unet.config.cross_attention_dim
        if name.startswith('mid_block'):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith('down_block'):
            block_id = int(name[len('down_blocks.')])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith('up_block'):
            block_id = int(name[len('up_blocks.')])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor() # We preverse the attention ops on self-attention layer
        else:
            layer_name_prefix = name.split('.processor')[0]
            weights = {
                'to_k_ip.weight': unet_state_dict[layer_name_prefix + 'to_k.weight'],
                'to_v_ip.weight': unet_state_dict[layer_name_prefix + 'to_v.weight']
            }
            # Assign IP-Adapter-based Attention mechanism
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_tokens=args.num_tokens
            )
            attn_procs[name].load_state_dict(weights) # init linear layers of new k, v from the existing ones respectively
    unet.set_attn_processor(attn_procs) # Reload attn processors with new ones

    # Load Ip-adapter pretrained weights
    ip_state_dict = torch.load(args.pretrained_ip_adapter_path, map_location='cpu')
    
    adapter_modules = torch.nn.ModuleList(unet.attention_processors.values())
    adapter_modules.load_state_dict(ip_state_dict['ip_adapter'], strict=True) # Load weights from Linear Layers (k, v) of pretrained IP-Adatper
    
    # Init projection layer for IP-Adapter
    # Here we use perceiver from Flamingo paper
    image_proj_model = Resampler(
        dim=image_encoder.config.hidden_size,
        depth=args.depth,
        dim_head=args.head_dim,
        heads=args.head_num,
        num_queries=args.num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
    )
    image_proj_model.load_state_dict(ip_state_dict['image_proj'], strict=True) # Load pretrained weights from pretrained IP-Adpater model

    # Update the first convolution layer to works with additional inputs
    new_in_channels = 13 # 4 (noisy image) + 4 (masked image) + 4 (denspose) + 1 (mask image)
    with torch.no_grad():
        conv_new = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=unet.conv_in.out_channels,
            kernel_size=3,
            padding=1,
        )
        conv_new.weight.data = conv_new.weight.data * 0. # Zero-initialized input
        conv_new.weight.data[:, unet.conv_in.in_channels, :, :] = unet.conv_in.weight.data # re-use conv weights for the original channels
        conv_new.bias.data = unet.conv_in.bias.data
        unet.conv_in = conv_new
        unet.config['in_channels'] = new_in_channels

    # Freeze some modules
    set_train(vae, False)
    set_train(text_encoder, False)
    set_train(image_encoder, False)
    
    # Trainable
    set_train(unet, True)
    set_train(image_proj_model, True)

    # Enable TF32 for faster training on Ampere GPUs (and later)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Define optimizer
    params_to_opt = itertools.chain(unet.parameters(), image_proj_model.parameters())
    optimizer = torch.optim.AdamW(
        params_to_opt,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    












    









