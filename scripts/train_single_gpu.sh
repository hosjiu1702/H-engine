CUDA_VISIBLE_VISIBLE="0" python -u -m accelerate.commands.launch \
train.py \
--pretrained_ip_adapter_path=checkpoints/ip-adapter-plus_sd15.bin \
--data_dir=datasets/vitonhd \
--output_dir=results \
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--allow_tf32 \
--use_densepose \
--mixed_precision=fp16 \
--num_workers=4 \
--num_train_epochs=4 \
--max_train_steps=12000 \
--checkpointing_steps=5000