export MIXED_PRECISION_TRAINING='bf16'
export NUM_GPUS=1
export NUM_NODES=1

CUDA_VISIBLE_VISIBLE="0" python -u -m accelerate.commands.launch --mixed_precision=$MIXED_PRECISION_TRAINING --num_processes=$NUM_GPUS --num_machines=$NUM_NODES --dynamo_backend='no' \
train_spacat.py \
--data_dir=datasets/vitonhd \
--use_subset \
--num_subset_samples=1000 \
--downscale \
--output_dir='results/spacat_minsnr' \
--train_batch_size=8 \
--gradient_accumulation_steps=1 \
--use_densepose \
--mixed_precision='bf16' \
--snr_gamma='5.0' \
--num_workers=4 \
--num_train_epochs=100 \
--max_train_steps=250000 \
--checkpointing_steps=10000 \
--validation_steps=10000 \
--lr=1e-5 \
--use_tracker=false \
--project_name=MinSNR-Traning
