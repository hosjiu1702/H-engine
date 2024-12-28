#/bin/bash
base_path=checkpoints/navier-1/navier-1-beta-1512

export MODEL_PATH="${base_path}/ckpt-60000 ${base_path}/ckpt-108000 ${base_path}/ckpt-156000 ${base_path}/ckpt-204000 ${base_path}/ckpt-252000"
export DATASET_NAME=dresscode
export DATAPATH=/hosjiu/data/DressCode
export OUTPUT_DIR=results/eval
export NUM_WORKERS=4
export BATCH_SIZE=4

python inference.py \
--model_path="$MODEL_PATH" \
--dataset_name=$DATASET_NAME \
--dresscode_datapath=$DATAPATH \
--output_dir=$OUTPUT_DIR \
--num_workers=$NUM_WORKERS \
--batch_size=$BATCH_SIZE \
--eval \
--save_metrics_to_file