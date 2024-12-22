CUDA_VISIBLE_VISIBLE="0" python -u -m accelerate.commands.launch --mixed_precision=fp16 --num_processes=1 --num_machines=1 --dynamo_backend='no' \
train_spacat.py \
--pretrained_ip_adapter_path=checkpoints/ip-adapter-plus_sd15.bin \
--data_dir=datasets/vitonhd \
--use_subset \
--num_subset_samples=1000 \
--downscale \
--output_dir=results \
--train_batch_size=4 \
--gradient_accumulation_steps=4 \
--use_densepose \
--mixed_precision=fp16 \
--num_workers=8 \
--num_train_epochs=100 \
--max_train_steps=250000 \
--checkpointing_steps=10000 \
--validation_steps=10000 \
--lr=1e-5 \
--use_tracker=true \
--project_name=Full_ft_IPAdapter_UNet
