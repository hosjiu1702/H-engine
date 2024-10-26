CUDA_VISIBLE_VISIBLE="0" accelerate launch \
train.py \
--pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-inpainting \
--pretrained_ip_adapter_path= \
--data_dir=datasets/vitonhd \
--output_dir=results \
--train_batch_size=4 \
--gradient_accumulation_steps=2 \
--num_workers=4 \
--num_train_epochs=4 \
--max_train_steps=12000 \
--checkpointing_steps=5000