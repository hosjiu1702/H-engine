export TORCH_DISTRIBUTED_DEBUG=INFO
export MIXED_PRECISION_TRAINING='bf16'
export NUM_GPUS=4
export NUM_NODES=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u -m accelerate.commands.launch --multi_gpu --mixed_precision=$MIXED_PRECISION_TRAINING --num_processes=$NUM_GPUS --num_machines=$NUM_NODES --dynamo_backend='no' \
train_spacat.py \
--data_dir=datasets/vitonhd \
--downscale \
--use_dilated_mask \
--snr_gamma='5.0' \
--use_densepose \
--output_dir='results/navier1-1' \
--train_batch_size=8 \
--gradient_accumulation_steps=1 \
--mixed_precision=$MIXED_PRECISION_TRAINING \
--num_workers=16 \
--num_train_epochs=150 \
--max_train_steps=290000 \
--checkpointing_steps=30000 \
--validation_steps=10000 \
--lr=1e-5 \
--use_tracker=true \
--save \
--project_name=Finetune-VTO
