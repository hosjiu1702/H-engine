#/bin/bash
base_path=/home/jupyter/H-engine/results/navier-1-1512

export MODEL_PATH="${base_path}/ckpt-312000"
export DATASET_NAME=vitonhd
export DATAPATH=datasets/vitonhd
export OUTPUT_DIR=results/eval
export NUM_WORKERS=8
export BATCH_SIZE=16
export DEVICE="cuda:0"

python inference.py \
--model_path="$MODEL_PATH" \
--dataset_name=$DATASET_NAME \
--vitonhd_datapath=$DATAPATH \
--output_dir=$OUTPUT_DIR \
--num_workers=$NUM_WORKERS \
--batch_size=$BATCH_SIZE \
--device=$DEVICE \
--eval
# --save_metrics_to_file