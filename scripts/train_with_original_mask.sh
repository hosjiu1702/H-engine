export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=2,3

python -u -m accelerate.commands.launch --multi_gpu --main_process_port 29400 --mixed_precision=fp16 --num_processes=2 --num_machines=1 --dynamo_backend='no' \
train_spacat.py \
--data_dir=datasets/vitonhd \
--use_subset \
--num_subset_samples=5000 \
--downscale \
--use_densepose \
--output_dir=results/original_mask \
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--mixed_precision=fp16 \
--num_workers=16 \
--num_train_epochs=240 \
--max_train_steps=300000 \
--checkpointing_steps=50000 \
--validation_steps=25000 \
--lr=1e-5 \
--use_tracker=true \
--save \
--project_name=Mask_Strategy