# CAT-VTON like training strategy
#   * We concatenate garment to person in spatial dim
#
# Modified or got inspired from:
#
# - https://github.com/miccunifi/ladi-vton/blob/master/src/train_vto.py
# - https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py
# - https://github.com/yisol/IDM-VTON/blob/1b39608bf3b6f075b21562e86302dcefd6989fc5/train_xl.py
# - https://github.com/lyc0929/OOTDiffusion-train/blob/main/run/ootd_train.py
# - https://github.com/luxiaolili/IDM-VTON-train/blob/main/train.py
# - https://github.dev/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
# - https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

import itertools
import math
import os
from os import path as osp
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from diffusers.utils import is_wandb_available
from accelerate import Accelerator, PartialState
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import bitsandbytes as bnb
from cleanfid import fid

from src.utils import (
    set_seed,
    set_train,
    use_gradient_accumulation,
    total_trainable_params,
    generate_rand_chars,
    str2bool,
    init_attn_processor,
    make_custom_stats
)
from src.models.denoising_unet import UNet2DConditionModel
from src.models.attention_processor import SkipAttnProcessor
from src.models.autoencoder_kl import AutoencoderKLForEmasc
from src.models.pme import PriorModelEvolution
from src.models.poseguider import PoseGuider
from src.models.reference_net import ReferenceNet
from src.models.reference_encoder import ReferenceEncoder
from src.models.reference_net_attention import ReferenceNetAttention
from src.pipelines.refnet_pipeline import TryOnPipeline
from src.dataset.vitonhd import VITONHDDataset
from src.dataset.dresscode import DressCodeDataset
from src.utils.training_utils import compute_snr



if is_wandb_available():
    import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument(
        '--pretrained_model_name_or_path',
        type=str,
        default='stable-diffusion-v1-5/stable-diffusion-inpainting',
        required=False,
        help='Path to pretrained model or model identifier from huggingface.co/models'
    )
    parser.add_argument(
        '--refnet_model',
        type=str,
        default='stable-diffusion-v1-5/stable-diffusion-inpainting',
        required=False,
        help='Path to pretrained model or model identifier from huggingface.co/models'
    )
    parser.add_argument(
        '--pretrained_ip_adapter_path',
        type=str,
        default='checkpoints/ip-adapter-plus_sd15.bin',
        required=False,
        help='Path to the pretrained IP-Adapter.'
    )
    parser.add_argument(
        '--image_encoder_path',
        type=str,
        default='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        required=False,
    )
    parser.add_argument(
        '--vae_path',
        type=str,
        default='stabilityai/sd-vae-ft-mse'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='The output directory where the checkpoints will be written.'
    )
    parser.add_argument(
        '--logging_dir',
        type=str,
        default='logs'
    )
    parser.add_argument(
        '--merge_hd_dc',
        action='store_true',
        help='Merge VITON-HD and DressCode to train as an unified training dataset.'
    )
    parser.add_argument(
        '--vitonhd_datapath',
        type=str,
        help='Path to root path of VITON-HD dataset.'
    )
    parser.add_argument(
        '--dresscode_datapath',
        type=str,
        help='Path to root path of DressCode dataset.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Path to the dataset which is used to train model.'
    )
    parser.add_argument(
        '--use_subset',
        action='store_true'
    )
    parser.add_argument(
        '--num_subset_samples',
        type=int,
        default=10
    ) 
    parser.add_argument(
        '--downscale',
        action='store_true',
        help='where or not to downscale all of images in the provided dataset.'
    )
    parser.add_argument(
        '--total_limit_states',
        type=int,
        default=3,
        help='the maximum value which control the number of saved model state dict'
    )
    parser.add_argument(
        '--use_tracker',
        type=str,
        default='true',
        help='Whether or not to use tracker to track experiments.'
    )
    parser.add_argument(
        '--report_to',
        type=str,
        default='wandb',
        help=('The tracker is used to report the results and logs to. '
              'Supported platforms are `"tensorboard"`, `"wandb"` (Default)'
        )
    )
    parser.add_argument(
        '--project_name',
        type=str,
        default='Finetune inpainting UNet',
        help='Project name for init with wandb'
    )
    parser.add_argument(
        '--wandb_name_run',
        type=str,
        default=None,
        help='Wandb name run.'
    )
    parser.add_argument(
        '--mixed_precision',
        type=str,
        default=None,
        choices=['no', 'fp16', 'bf16'],
        help='Whether to use mixed precision training.'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of update steps to accumulate before performing a backward pass.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1996,
        help='A seed for reproducible training'
    )
    parser.add_argument(
        '--allow_tf32',
        action='store_true',
        help='Whether or not to allow TF32 on Ampere GPUs.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        help='Initial learning rate (after the potential warmup period) to use.'
    )
    parser.add_argument(
        '--adam_beta1',
        type=float,
        default=0.9,
        help='The beta 1 parameter for the Adam optimizer.'
    )
    parser.add_argument(
        '--adam_beta2',
        type=float,
        default=0.999,
        help='The beta 2 parameter for the Adam optimizer.'
    )
    parser.add_argument(
        '--adam_epsilon',
        type=float,
        default=1e-8,
        help='Epsilon value for the Adam optimzer.'
    )
    parser.add_argument(
        '--adam_weight_decay',
        type=float,
        default=1e-2,
        help='Weight decay to use.'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=1024,
        help='Height of the generated image.'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=768,
        help='Width of the generated image.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=8,
        help='Batch size (per device, in the distributed training) for the training dataloader.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of workers to use in the dataloader.'
    )
    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=10000,
        help='Total number of training steps to perform. If provided, overrides num_train_epochs.'
    )
    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=100,
        help='Total number of traning epochs to perform.'
    )
    parser.add_argument(
        '--checkpointing_steps',
        type=int,
        default=2500,
        help=('Save a checkpoint of the training state every X updates. '
              'These checkpoints are only suitable for resuming to continue training.'
        )
    )
    parser.add_argument(
        '--validation_steps',
        type=int,
        default=1000,
        help='Run validation every X steps.'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=10000,
        help='Run FID evaluation every X steps.'
    )
    parser.add_argument(
        '--use_densepose',
        action='store_true',
        help='Whether or not use densepose alongside with (mask, agnostic image and original image)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Whether or not to save model after each validation step.'
    )
    parser.add_argument(
        '--use_dilated_mask',
        action='store_true',
        help='Whether or not to use Dilated-relaxed Mask in section 3.3 in FitDit paper'
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        '--progressive_training',
        action='store_true',
        help="Do progressive training by resuming from a given checkpoint then continue training with high resolution images." \
        "Paper reference:" \
        "   - Learning Flow Fields in Attention for Controllable Person Image Generation (https://arxiv.org/abs/2412.08486v2)." \
        "   - M&M VTO: Multi-Garment Virtual Try-On and Editing (https://arxiv.org/abs/2406.04542)."
    )
    parser.add_argument(
        '--training_state_path',
        type=str,
        help='Path to the model\'s state that you want to resume aiming to do progressive training.'
    )
    parser.add_argument(
        '--train_with_8bit',
        action='store_true'
    )
    parser.add_argument(
        '--train_self_attn_only',
        action='store_true',
        help='Train only self-attention layers in UNet.'
    )
    parser.add_argument(
        '--train_refnet_self_attn_only',
        action='store_true',
    )
    parser.add_argument(
        '--dataset_augmentation',
        action='store_true',
        help='Apply some augmentation transformation aiming to achieve better generalization in real-world scenarios. Checkout section Training & Inference Details in Supplementary Material from StableVITON (https://arxiv.org/pdf/2312.01725)'
    )
    parser.add_argument(
        '--random_dilate_mask',
        action='store_true',
        help='This only must be applied to rigorous rectangular mask shape (v2). Checkout section 3.3 of https://arxiv.org/abs/2411.10499'
    )
    parser.add_argument(
        '--prior_model_evolution',
        action='store_true',
        help='Apply Prior Model Evolution as explained in https://arxiv.org/abs/2405.18172'
    )
    parser.add_argument(
        '--hd',
        action='store_true',
        help='Single VITON-HD training'
    )
    parser.add_argument(
        '--dc',
        action='store_true',
        help='Single DRESSCODE training'
    )
    parser.add_argument(
        '--cfg',
        default=1.,
        type=float,
    )
    parser.add_argument(
        '--enable_resampler',
        action='store_true',
        help='Add a resampler to reference encoder as mentioned in IP-Adapter.'
    )
    parser.add_argument(
        '--enable_mlp',
        action='store_true',
    )
    parser.add_argument(
        '--train_resampler',
        action='store_true'
    )
    parser.add_argument(
        '--train_mlp',
        action='store_true'
    )
    parser.add_argument(
        '--train_midup_self_attn',
        action='store_true'
    )
    
    args = parser.parse_args()

    return args


class UnifiedModel(nn.Module):
    def __init__(self, unet, refnet, poseguider, reference_encoder, reference_control_writer, reference_control_reader):
        super().__init__()
        self.unet = unet
        self.refnet = refnet
        self.reference_encoder = reference_encoder
        self.poseguider = poseguider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
    
    def forward(self, noisy_latents, masks, masked_image_latents, cloth_latents, timesteps, pose_images, cloth_ref_images):
        ref_timesteps = torch.zeros_like(timesteps)
        pose_latents = self.poseguider(pose_images)
        noisy_latents = noisy_latents + pose_latents
        latent_model_input = torch.cat([noisy_latents, masks, masked_image_latents], dim=1)

        encoder_hidden_states = self.reference_encoder(cloth_ref_images)
        self.refnet(cloth_latents, ref_timesteps, encoder_hidden_states)
        self.reference_control_reader.update(self.reference_control_writer)
        noise_pred = self.unet(latent_model_input, timesteps, encoder_hidden_states).sample
        
        return noise_pred


def collate_fn(data):
    images = torch.stack([sample['image'] for sample in data])
    masked_images = torch.stack([sample['masked_image'] for sample in data])
    masks = torch.stack([sample['mask'] for sample in data])
    denseposes = torch.stack([sample['densepose'] for sample in data])
    cloth_raw = torch.stack([sample['cloth_raw'] for sample in data])
    cloth_ref = torch.cat([sample['cloth_ref'] for sample in data], dim=0)
    drop_image_embeds = torch.Tensor([sample['drop_image_embed'] for sample in data])
    
    return {
        'image': images,
        'masked_image': masked_images,
        'mask': masks,
        'densepose': denseposes,
        'cloth_raw': cloth_raw,
        'cloth_ref': cloth_ref,
        'drop_image_embed': drop_image_embeds
    }
    
    
def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    # Init Acclerator object for tracking training process
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
        total_limit=args.total_limit_states, # this argument should control the number of saved model's state dict
    )
    accelerator = Accelerator(
                mixed_precision=args.mixed_precision,
                log_with=args.report_to,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                project_config=project_config,
                kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
            )
    device = accelerator.device

    # For reproducibility
    if args.seed:
        set_seed(args.seed)

    # Load diffusion-related components
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler', rescale_betas_zero_snr=False)
    vae = AutoencoderKL.from_pretrained(args.vae_path, use_safetensors=True)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet', use_safetensors=False)
    refnet = ReferenceNet.from_pretrained(args.refnet_model, subfolder='unet', use_safetensors=False)
    poseguider = PoseGuider(noise_latent_channels=4)
    reference_encoder = ReferenceEncoder(args.image_encoder_path, enable_resampler=args.enable_resampler, enable_mlp=args.enable_mlp, unet=unet)

    modules = [vae, unet, refnet, poseguider, reference_encoder]

    reference_control_writer = ReferenceNetAttention(refnet, mode='write', fusion_blocks='full')
    reference_control_reader = ReferenceNetAttention(unet, mode='read', fusion_blocks='full')
    
    # init_attn_processor(unet, cross_attn_cls=SkipAttnProcessor) # skip cross-attention layer
    # init_attn_processor(refnet, cross_attn_cls=SkipAttnProcessor)
    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
    unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)

    if unet.conv_in.in_channels == 4:
        raise RuntimeError('This script supports inpainting UNet only.')

    # if args.use_densepose:
    #     new_in_channels = 13 # 4 (noisy image) + 4 (masked image) + 4 (denspose) + 1 (mask image) -- spatial dimension concatenation
    #     # Add some channels to the first input convolution layer
    #     # if we use additional auxiliary inputs like densepose, skeleton,...
    #     with torch.no_grad():
    #         conv_new = torch.nn.Conv2d(
    #             in_channels=new_in_channels,
    #             out_channels=unet.conv_in.out_channels,
    #             kernel_size=3,
    #             padding=1,
    #         )
    #         conv_new.weight.data = conv_new.weight.data * 0. # Zero-initialized input
    #         conv_new.weight.data[:, :unet.conv_in.in_channels, :, :] = unet.conv_in.weight.data # re-use conv weights for the original channels
    #         conv_new.bias.data = unet.conv_in.bias.data
    #         unet.conv_in = conv_new
    #         unet.config['in_channels'] = new_in_channels
    #         unet.config.in_channels = new_in_channels

    set_train(vae, False)
    set_train(reference_encoder, False)
    set_train(refnet, True)
    set_train(poseguider, True)
    set_train(unet, True)
    
    if args.train_self_attn_only:
        set_train(unet, False)
        for name, module in unet.named_modules():
            if name.endswith('.attn1'):
                for params in module.parameters():
                    params.requires_grad = True

    if args.train_midup_self_attn:
        set_train(unet, False)
        for name, module in unet.named_modules():
            if name.endswith('.attn1') and not name.startswith('down_blocks'):
                for params in module.parameters():
                    params.requires_grad = True

    if args.train_refnet_self_attn_only:
        set_train(refnet, False)
        for name, module in refnet.named_modules():
            if name.endswith('.attn1'):
                for params in module.parameters():
                    params.requires_grad = True
    
    if args.enable_resampler and args.train_resampler:
        set_train(reference_encoder, False)
        set_train(reference_encoder.image_proj_model, True)

    if args.enable_mlp and args.train_mlp:
        set_train(reference_encoder, False)
        set_train(reference_encoder.mlp, True)
    # else:
    #     # train full unet
    #     set_train(unet, True) # train full unet

    # if args.prior_model_evolution:
    #     model_evolver = PriorModelEvolution(device=unet.device.type)
    #     model_evolver(unet)
    #     if unet.device.type == 'cuda':
    #         del model_evolver
    #         torch.cuda.empty_cache()

    accelerator.print('\n==== Trainable Params ====')
    trainable_params = 0
    for m in modules:
        s = total_trainable_params(m)
        if s > 0:
            accelerator.print(f'* {type(m).__name__}: {s}')
            trainable_params += s
    accelerator.print('=====================')
    accelerator.print(f'Total: {trainable_params}')
    
    # Enable TF32 for faster training on Ampere GPUs (and later)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.train_with_8bit:
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    model = UnifiedModel(
        unet=unet, refnet=refnet, poseguider=poseguider,
        reference_encoder=reference_encoder,
        reference_control_writer=reference_control_writer,
        reference_control_reader=reference_control_reader
    )
        
    # Define optimizer
    params_to_opt = itertools.chain(
        model.unet.parameters(),
        model.poseguider.parameters(),
        model.refnet.parameters(),
        model.reference_encoder.mlp.parameters()
    )
    optimizer = optimizer_class(
        params_to_opt,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    if args.downscale:
        vars(args)['height'] = args.height // 2
        vars(args)['width'] = args.width // 2

    if args.merge_hd_dc:
        # VITON-HD
        hd_train_dataset = VITONHDDataset(
            data_rootpath=args.vitonhd_datapath,
            use_trainset=True,
            height=args.height,
            width=args.width,
            use_dilated_relaxed_mask=True if args.use_dilated_mask else False,
            use_augmentation=True if args.dataset_augmentation else False,
            random_dilate_mask=True if args.random_dilate_mask else False
        )
        hd_test_dataset = VITONHDDataset(
            data_rootpath=args.vitonhd_datapath,
            use_trainset=False,
            height=args.height,
            width=args.width,
            use_dilated_relaxed_mask=True if args.use_dilated_mask else False,
        )
        # DRESSCODE
        dc_train_dataset = DressCodeDataset(
            args.dresscode_datapath,
            phase='train',
            h=args.height,
            w=args.width,
            use_dilated_relaxed_mask=True if args.use_dilated_mask else False,
            use_augmentation=True if args.dataset_augmentation else False,
            random_dilate_mask=True if args.random_dilate_mask else False
        )
        dc_test_dataset = DressCodeDataset(
            args.dresscode_datapath,
            phase='test',
            h=args.height,
            w=args.width,
            use_dilated_relaxed_mask=True if args.use_dilated_mask else False,
        )
        train_dataset = ConcatDataset([hd_train_dataset, dc_train_dataset])
        test_dataset = ConcatDataset([hd_test_dataset, dc_test_dataset])
    else:
        if args.hd:
            train_dataset = VITONHDDataset(
                data_rootpath=args.vitonhd_datapath,
                use_trainset=True,
                use_paired_data=True,
                height=args.height,
                width=args.width,
                use_CLIPVision=True,
                use_dilated_relaxed_mask=True if args.use_dilated_mask else False,
                use_augmentation=True if args.dataset_augmentation else False,
                random_dilate_mask=True if args.random_dilate_mask else False
            )
            test_dataset = VITONHDDataset(
                data_rootpath=args.vitonhd_datapath,
                use_trainset=False,
                height=args.height,
                width=args.width,
                use_dilated_relaxed_mask=True if args.use_dilated_mask else False,
            )
        elif args.dc:
            train_dataset = DressCodeDataset(
                args.dresscode_datapath,
                phase='train',
                h=args.height,
                w=args.width,
                use_dilated_relaxed_mask=True if args.use_dilated_mask else False,
                use_augmentation=True if args.dataset_augmentation else False,
                random_dilate_mask=True if args.random_dilate_mask else False
            )
            test_dataset = DressCodeDataset(
                args.dresscode_datapath,
                phase='test',
                h=args.height,
                w=args.width,
                use_dilated_relaxed_mask=True if args.use_dilated_mask else False,
            )
        else:
            raise ValueError(f'No support for your dataset.')

    if args.use_subset:
        t = args.num_subset_samples // 2
        sample_indexes = [i for i in range(t)] + [-i for i in range(1, t + 1)]
        train_dataset = Subset(train_dataset, sample_indexes)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # For mixed precision training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    # cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    vae.to(device, dtype=weight_dtype)

    # hey Huggingface team, why do you want to need to override this poor variable (overrode...)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = num_update_steps_per_epoch * args.num_train_epochs
        overrode_max_train_steps = True

    # Set up everything for:
    # * Distributed training
    # * Data Parallelism
    # * Device Placement
    # * Gradient Synchronization
    # * what else?
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # (actually I don't know why this could happen :|)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize tracker, store the configuration
    if str2bool(args.use_tracker):
        if accelerator.is_main_process:
            accelerator.init_trackers(
                project_name=args.project_name,
                config=dict(vars(args)),
                init_kwargs={'wandb': {'name': args.wandb_name_run}} if args.wandb_name_run else {}
            )

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("\n********* Running Training *********")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    accelerator.print("********* Running Training *********\n")

    global_steps = 0
    start_epoch = 0
    if args.progressive_training:
        try:
            accelerator.load_state(args.training_state_path)
            state_name = os.path.basename(args.training_state_path)
            global_steps = int(state_name.split('-')[0])
            start_epoch = global_steps // num_update_steps_per_epoch
            resume_step = global_steps % num_update_steps_per_epoch
        except ValueError as e:
            global_steps = int(state_name.split('-')[1])
            start_epoch = global_steps // num_update_steps_per_epoch
            resume_step = global_steps % num_update_steps_per_epoch

    progress_bar = tqdm(
        range(start_epoch, args.max_train_steps),
        desc='Steps',
        disable=not accelerator.is_local_main_process, # Only show the progress bar once on each machine.
    )

    test_batch = next(iter(test_dataloader))
    for epoch in range(start_epoch, args.num_train_epochs):
        unet.train()
        refnet.train()
        poseguider.train()
        reference_encoder.mlp.train()

        train_loss = 0.
        for step, batch in enumerate(train_dataloader):
            if args.progressive_training and epoch == start_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(unet), accelerator.accumulate(poseguider), accelerator.accumulate(refnet):
                # Get inputs for denoising unet (Pixel Space --> Latent Space)
                image_latents = vae.encode(batch['image'].to(dtype=weight_dtype)).latent_dist.sample()
                image_latents = image_latents * vae.config.scaling_factor
                masked_image_latents = vae.encode(batch['masked_image'].to(dtype=weight_dtype)).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor
                densepose_latents = vae.encode(batch['densepose'].to(dtype=weight_dtype)).latent_dist.sample()
                densepose_latents = densepose_latents * vae.config.scaling_factor
                cloth_latents = vae.encode(batch['cloth_raw'].to(dtype=weight_dtype)).latent_dist.sample()
                cloth_latents = cloth_latents * vae.config.scaling_factor

                masks = batch['mask'].to(dtype=weight_dtype)
                masks = F.interpolate(masks, size=(args.height//8, args.width//8))

                # classifier-free guidance support
                image_ref = []
                for img_ref, drop in zip(batch['cloth_ref'], batch['drop_image_embed']):
                    if drop == 1:
                        image_ref.append(torch.zeros_like(img_ref)) # shape = [3, 224, 224]
                    else:
                        image_ref.append(img_ref)
                image_ref = torch.stack(image_ref) # shape = [bs, 3, 224, 224]
                assert image_ref.shape[0] == args.train_batch_size

                # Move to device (e.g, GPUs)
                image_latents.to(device, dtype=weight_dtype)
                masked_image_latents.to(device, dtype=weight_dtype)
                densepose_latents.to(device, dtype=weight_dtype)
                masks.to(device, dtype=weight_dtype)
                image_ref.to(device, dtype=weight_dtype)

                # Add Gaussian noise to input for each timestep
                # this is diffusion forward pass
                noise = torch.randn_like(masked_image_latents).to(device, dtype=weight_dtype)
                bs = image_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device) # DDIM Scheduler
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(image_latents, noise, timesteps)
                noise_pred = model(noisy_latents, masks, masked_image_latents, cloth_latents, timesteps, batch['densepose'], image_ref)
#                 encoder_hidden_states = image_encoder(image_ref).unsqueeze(1)

#                 # Get self-attention features from reference net
#                 # and inject it into denoising unet
#                 ref_timesteps = torch.zeros_like(timesteps) # why zero out noise?
#                 refnet(cloth_latents, ref_timesteps, encoder_hidden_states)
#                 reference_control_reader.update(reference_control_writer)

#                 # Denoising unet
#                 noisy_latents = noise_scheduler.add_noise(image_latents, noise, timesteps)
#                 concat_latents = torch.cat([noisy_latents, masks, masked_image_latents], dim=1) # concatenate in channel dim
#                 pose_latents = poseguider(batch['densepose'].to(dtype=weight_dtype))
#                 unet_input = concat_latents + pose_latents
#                 noise_pred = unet(unet_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample # Denoising or diffusion backward process

                if args.snr_gamma is None:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean') # compute loss
                else:
                    snr_timesteps = timesteps
                    snr = compute_snr(noise_scheduler, snr_timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(snr_timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    assert noise_scheduler.config.prediction_type == 'epsilon', "Only support noise prediction for Min-SNR weighting strategy."
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # For logging purpose, you need to gather losses across
                # all processes (in distributed training if any) manually
                # I'am not sure but IMO they probably use DDP (Distributed Data Parallel)
                # behind the scene so feel free check it out if you want to learn more on this concept
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean() # this line from Huggingface Accelerate is suck :)
                if not use_gradient_accumulation(args.gradient_accumulation_steps):
                    train_loss += avg_loss.item()
                else:
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Huggingface Accelerate seems do almost everything here
                # so it obscures too much things under the hood making it
                # very hard to understand how everything is going on
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(params_to_opt, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                reference_control_writer.clear()
                reference_control_reader.clear()
                
                # Logging training loss after updating all network weights
                if accelerator.sync_gradients:
                    global_steps += 1
                    progress_bar.update(1)
                    accelerator.log({'train_loss': train_loss}, step=global_steps) # log to predefined tracker, for example, wandb
                    train_loss = 0.
                    if accelerator.is_main_process:
                        # Saves model's state at a certain training step
                        if global_steps % args.checkpointing_steps == 0:
                            if args.save:
                                # Just for resuming when we want to continue training from the last state
                                save_path = os.path.join(args.output_dir, f'state/{global_steps}-steps') # should be added a timestamp
                                os.makedirs(save_path, exist_ok=True)
                                accelerator.save_state(save_path, safe_serialization=False)
                                accelerator.print(f'Saved state to {save_path}')
                        if global_steps % args.eval_steps == 0:
                            # FID Evaluation
                            unwrapped_unet = accelerator.unwrap_model(unet)
                            with torch.no_grad():
                                """ Init temporarily pipeline for inferencing."""
                                pipe = TryOnPipeline(
                                    vae=vae,
                                    unet=unwrapped_unet,
                                    scheduler=noise_scheduler,
                                ).to(device)

                                test_set = DressCodeDataset(
                                    args.dresscode_datapath,
                                    phase='test',
                                    order='paired',
                                    h=args.height,
                                    w=args.width,
                                    use_dilated_relaxed_mask=True
                                )
                                test_dataloader = DataLoader(
                                    test_set,
                                    batch_size=torch.cuda.device_count(),
                                    shuffle=False,
                                    num_workers=16,
                                    pin_memory=True
                                )

                                fid_dir = "/tmp/fid/"
                                os.makedirs(fid_dir, exist_ok=True)

                                distributed_state = PartialState()
                                pipe.to(distributed_state.device)

                                for _, batch in enumerate(tqdm(test_dataloader)):
                                    with distributed_state.split_between_processes(batch) as input:
                                        with torch.amp.autocast(accelerator.device.type):
                                            images = pipe(
                                                image=input['image'].to(accelerator.device.type),
                                                mask_image=input['mask'].to(accelerator.device.type),
                                                densepose_image=input['densepose'].to(accelerator.device.type),
                                                cloth_image=input['cloth_raw'].to(accelerator.device.type),
                                                height=args.height,
                                                width=args.width,
                                                guidance_scale=1.5,
                                                num_inference_steps=30
                                            ).images
                                            for img, name in zip(images, batch['im_name']):
                                                img.save(osp.join(fid_dir, f'{name}'), quality=100, subsampling=0)

                                if not fid.test_stats_exists(name='dresscode', mode='clean'):
                                    # makes dataset statistics (features from InceptionNet-v3 by default)
                                    make_custom_stats(dataset_name='dresscode', dataset_path=args.dresscode_datapath)
                                fid_score = fid.compute_fid(
                                    fdir1=fid_dir,
                                    dataset_name='dresscode',
                                    mode='clean',
                                    dataset_split='custom',
                                    verbose=True,
                                    use_dataparallel=False
                                )
                                fid_score = round(fid_score, 3)
                                tracker = accelerator.get_tracker('wandb')
                                tracker.log({'fid (Dresscode)': fid_score})
                                del unwrapped_unet
                                del pipe
                                torch.cuda.empty_cache()
                        if global_steps % args.validation_steps == 0:
                            unwrapped_unet = accelerator.unwrap_model(unet)
                            unwrapped_refnet = accelerator.unwrap_model(refnet)
                            unwrapped_poseguider = accelerator.unwrap_model(poseguider)
                            unwrapped_encoder = accelerator.unwrap_model(reference_encoder)
                            pipe = TryOnPipeline(
                                unet=unwrapped_unet,
                                refnet=unwrapped_refnet,
                                poseguider=unwrapped_poseguider,
                                reference_encoder=unwrapped_encoder,
                                reference_control_writer=reference_control_writer,
                                reference_control_reader=reference_control_reader,
                                vae=vae,
                                scheduler=noise_scheduler,
                            )
                            with torch.no_grad():
                                # allows to run in mixed precision mode
                                # not using in backward pass
                                with torch.amp.autocast(device.type):
                                    # Generates some try-on results
                                    batch = test_batch
                                    images = pipe(
                                        image=batch['image'].to(device.type, dtype=weight_dtype),
                                        mask_image=batch['mask'].to(device.type, dtype=weight_dtype),
                                        densepose_image=batch['densepose'].to(device.type, dtype=weight_dtype),
                                        cloth_image=batch['cloth_raw'].to(device.type, dtype=weight_dtype),
                                        cloth_ref_image=batch['cloth_ref'].squeeze(1).to(device.type, dtype=weight_dtype),
                                        height=args.height,
                                        width=args.width,
                                        guidance_scale=args.cfg,
                                        num_inference_steps=35
                                    ).images # pil
                                    img_path = os.path.join(args.output_dir, 'images')
                                    os.makedirs(img_path, exist_ok=True)
                                    if args.report_to == 'wandb' and str2bool(args.use_tracker):
                                        wandb_tracker = accelerator.get_tracker('wandb')
                                        # concate generated image and original image for comparison
                                        results = []
                                        for img, origin_img, im_name in zip(images, batch['original_image'], batch['im_name']):
                                            index = im_name.split('.')[0]
                                            origin_img = to_pil_image(origin_img)
                                            output_img = Image.new('RGB', (img.width * 2, img.height))
                                            output_img.paste(img, (0, 0))
                                            output_img.paste(origin_img, (img.width, 0))
                                            results.append(wandb.Image(output_img, caption=f'index: {index}'))
                                            # save every generated images in the training batch into disk
                                            img.save(os.path.join(img_path, im_name))
                                        wandb_tracker.log({
                                            'validation': results
                                        })
                            # save full pipeline
                            if args.save:
                                save_path = os.path.join(args.output_dir, f'checkpoint/{global_steps}-steps')
                                pipe.save_pretrained(save_path)
                            del unwrapped_unet
                            del unwrapped_refnet
                            del unwrapped_poseguider
                            del unwrapped_encoder
                            del pipe
                            torch.cuda.empty_cache()
                logs = {'step_loss': loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_steps >= args.max_train_steps:
                    break

    accelerator.end_training()
    accelerator.print('======== Training End ========')


if __name__ == '__main__':
    main()
