export TORCH_DISTRIBUTED_DEBUG=INFO
export MIXED_PRECISION_TRAINING='bf16'
export NUM_GPUS=4
export NUM_NODES=1
export MAIN_PROCESS_PORT=29505
export DEVICE=0,1,2,3
export SNR_GAMMA=3
export VITONHD_DATAPATH=datasets/vitonhd
export DRESSCODE_DATAPATH=datasets/dresscode
export OUTPUT_DIR=results/navier-1-1512-1024x768
export PROJECT_NAME='Finetune-VTO'
export WANDB_NAME_RUN='Navier-1[Beta].1512[1024x768]'
export ENABLE_TRACKER=true

CUDA_VISIBLE_DEVICES=$DEVICE python -u -m accelerate.commands.launch --multi_gpu --main_process_port=$MAIN_PROCESS_PORT --mixed_precision=$MIXED_PRECISION_TRAINING --num_processes=$NUM_GPUS --num_machines=$NUM_NODES --dynamo_backend='no' \
train_spacat.py \
--merge_hd_dc \
--vitonhd_datapath=$VITONHD_DATAPATH \
--dresscode_datapath=$DRESSCODE_DATAPATH \
--use_dilated_mask \
--use_densepose \
--snr_gamma=$SNR_GAMMA \
--output_dir=$OUTPUT_DIR \
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--mixed_precision=$MIXED_PRECISION_TRAINING \
--num_workers=16 \
--num_train_epochs=100 \
--max_train_steps=500000 \
--checkpointing_steps=50000 \
--validation_steps=12000 \
--lr=1e-5 \
--use_tracker=$ENABLE_TRACKER \
--project_name=$PROJECT_NAME \
--wandb_name_run=$WANDB_NAME_RUN \
--save
