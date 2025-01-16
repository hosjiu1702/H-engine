#/bin/bash
base_path=results/navier-1-1512-1024x768

export MODEL_PATH="${base_path}/ckpt-96000 ${base_path}/ckpt-108000 ${base_path}/ckpt-120000 ${base_path}/ckpt-132000"
export DATASET_NAME=dresscode
export DATAPATH=datasets/dresscode
export OUTPUT_DIR=results/eval
export NUM_WORKERS=8
export BATCH_SIZE=4
export HEIGHT=1024
export WIDTH=768
export DEVICE="cuda:0"

python inference.py \
--model_path="$MODEL_PATH" \
--dataset_name=$DATASET_NAME \
--dresscode_datapath=$DATAPATH \
--height=$HEIGHT \
--width=$WIDTH \
--output_dir=$OUTPUT_DIR \
--num_workers=$NUM_WORKERS \
--batch_size=$BATCH_SIZE \
--device=$DEVICE \
--save_metrics_to_file