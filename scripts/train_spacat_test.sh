export TORCH_DISTRIBUTED_DEBUG=INFO
export MIXED_PRECISION_TRAINING='bf16'
export NUM_GPUS=1
export NUM_NODES=1
export MAIN_PROCESS_PORT=29505
export DEVICE=0
export SNR_GAMMA=2
export DATA_DIR=datasets/vitonhd
export PROJECT_NAME='Finetune-VTO'
export ENABLE_TRACKER=false

CUDA_VISIBLE_DEVICES=$DEVICE python -u -m accelerate.commands.launch --main_process_port=$MAIN_PROCESS_PORT --mixed_precision=$MIXED_PRECISION_TRAINING --num_processes=$NUM_GPUS --num_machines=$NUM_NODES --dynamo_backend='no' \
train_spacat.py \
--data_dir=$DATA_DIR \
--use_subset \
--num_subset_samples=1000 \
--downscale \
--use_dilated_mask \
--use_densepose \
--snr_gamma=$SNR_GAMMA \
--output_dir=results/min-snr/gamma-${SNR_GAMMA} \
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
--mixed_precision=$MIXED_PRECISION_TRAINING \
--num_workers=8 \
--num_train_epochs=100 \
--max_train_steps=250000 \
--checkpointing_steps=10000 \
--validation_steps=4 \
--lr=1e-5 \
--use_tracker=$ENABLE_TRACKER \
--project_name=$PROJECT_NAME \
--wandb_name_run=gamma-${SNR_GAMMA}
--save
