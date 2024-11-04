CUDA_VISIBLE_VISIBLE="0" python -u -m accelerate.commands.launch --mixed_precision=fp16 --num_processes=1 --num_machines=1 --dynamo_backend='no' \
train.py \
--pretrained_ip_adapter_path=checkpoints/ip-adapter-plus_sd15.bin \
--data_dir=datasets/vitonhd \
--use_subset \
--output_dir=results \
--train_batch_size=2 \
--gradient_accumulation_steps=1 \
--use_densepose \
--mixed_precision=fp16 \
--num_workers=8 \
--num_train_epochs=4 \
--max_train_steps=12000 \
--checkpointing_steps=5000 \
--validation_steps=6 \
--lr=1e-5 \
--project_name=Full_ft_IPAdapter_UNet