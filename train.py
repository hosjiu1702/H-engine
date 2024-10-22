# Modified from:
# - https://github.com/miccunifi/ladi-vton/blob/master/src/train_vto.py
# - https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py
# - https://github.com/yisol/IDM-VTON/blob/1b39608bf3b6f075b21562e86302dcefd6989fc5/train_xl.py

import itertools
import math
import os

import torch
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel
)
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import tqdm

from src.utils.utils import set_seed, set_train, use_gradient_accumulation
from src.models.ip_adapter.attention_processor import (
    AttnProcessor2_0 as AttnProcessor,
    IPAttnProcessor2_0 as IPAttnProcessor
)
from src.models.ip_adapter.resampler import Resampler


logger = get_logger(__name__, log_level="INFO")


def main():
    args = None # update later

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    # Init Acclerator object for tracking training process
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
        total_limit=args.total_limit_states,
    )
    accelerator = Accelerator(
                mixed_precision=args.mixed_precision,
                log_with=args.report_to,
                gradient_accumlation_steps=args.gradient_accumulation_steps,
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
    
    # Init image projection layer from outside of the denoising unet
    # for a better control because we could avoid directly modify the code inside of unet2dcondition.
    # But this will make this training code hard to read and refractor with new training strategy.
    unet.encoder_hid_proj = image_proj_model
    unet.config.update({'encoder_hidden_dim_type': 'ip_image_proj'})

    # Update the first convolution layer to works with additional inputs
    if args.use_densepose:
        new_in_channels = 13 # 4 (noisy image) + 4 (masked image) + 4 (denspose) + 1 (mask image)
    else:
        new_in_channels = 9
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
    
    # Update later
    train_dataloader = None
    test_dataloader = None

    # For mixed precision training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    # cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    device = accelerator.device
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)

    # hey Huggingface team, why do you want to need to override this poor variable (overrode...)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumlation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = num_update_steps_per_epoch * args.num_train_epochs
        overrode_max_train_steps = True

    # Set up everything for:
    # * Distributed training
    # * Data Parallelism
    # * Device Placement
    # * Gradient Synchronization
    # * what else?
    unet, image_proj_model, image_encoder, optimizer, train_dataloader = accelerator.prepare(
        unet,
        image_proj_model,
        image_encoder,
        optimizer,
        train_dataloader,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # (actually I don't know why this could happen :|)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize tracker, store the configuration
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("********* Running Training *********")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description('Steps')

    global_steps = 0
    for epoch in range(0, args.num_train_epochs):
        train_loss = 0.
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(image_proj_model):
                # Get inputs for denoising unet (Pixel Space --> Latent Space)
                latents = vae.encode(batch['image'].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                masked_images = vae.encode(batch['masked_image'].to(dtype=weight_dtype)).latent_dist.sample()
                masked_images = masked_images * vae.config.scaling_factor
                masks = batch['mask'].to(dtype=weight_dtype)
                masks = F.interpolate(masks, size=(args.height//8, args.weight//8))
                
                # Get text condition
                text_prompts = ['']*len(batch['captions']) # we set input text prompts as a list of empty strings
                text_ids = tokenizer(
                    text_prompts,
                    max_length=tokenizer.model_max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                ).input_ids
                encoder_hidden_states = text_encoder(text_ids.to(device)).last_hidden_state # use last feature layer from CLIP Text Encoder
                    
                image_embeds = image_encoder(batch['clip_image'].to(device, dtype=weight_dtype))
                ip_image_embeds = image_proj_model(image_embeds)
                added_cond_kwargs = {'image_embeds': ip_image_embeds}

                latents.to(device, dtype=weight_dtype)
                masked_images.to(device, dtype=weight_dtype)
                masks.to(device, dtype=weight_dtype)

                # Add Gaussian noise to input for each timestep
                # this is diffusion forward pass
                noise = torch.randn_like(latents).to(device, dtype=weight_dtype)
                bs = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_steps, (bs,))
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                unet_input = torch.cat([noisy_latents, masks, masked_images], dim=1) # concatenate in channel dim
                noise_pred = unet(unet_input, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs).sample # Denoising or diffusion backward process
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean') # compute loss

                # For logging purpose, you need to gather losses across
                # all processes (in distributed training if any) manually
                # I'am not sure but IMO they probably use DDP (Distributed Data Parallel)
                # behind the scene so feel free check it out if you want to learn more on this concept
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size))
                if not use_gradient_accumulation(args.gradient_accumlation_steps):
                    train_loss += avg_loss.item()
                else:
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Huggingface Accelerate seems do almost everything here
                # so it obscures too much things under the hood making it
                # very hard to understand how everything is going on
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                # Logging training loss after updating all network weights
                if accelerator.sync_gradients:
                    global_steps += 1
                    progress_bar.update(1)
                    accelerator.log({'train_loss': train_loss}, step=global_steps)
                    train_loss = 0.
                    # Saves model at a certain training step
                    if global_steps % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # Just for resuming when we want to continue training from the last state
                            save_path = os.path.join(args.output_dir, 'checkpoints', f'ckpt-{global_steps}')
                            os.makeddirs(save_path, exist_ok=True)
                            accelerator.save_state(save_path, safe_serialization=False)
                            logger.info(f'Saved state to {save_path}')
                            # Save (unet + ip-adapter)
                            unwrapped_unet = accelerator.unwrap_model(unet)
                            unwrapped_ipadapter = accelerator.unwrap_model(image_proj_model)
                            unet_path = os.path.join(args.output_dir, f'unet-{global_steps}.pth')
                            ipadapter_path = os.path.join(args.output_dir, f'ipadapter-{global_steps}.pth')
                            accelerator.save(unwrapped_unet, unet_path, safe_serialization=False)
                            accelerator.save(unwrapped_ipadapter, ipadapter_path, safe_serialization=False)
                            del unwrapped_unet
                            del unwrapped_ipadapter
 
                if global_steps >= args.max_train_steps:
                    break




















