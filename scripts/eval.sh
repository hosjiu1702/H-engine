export MODEL_PATH=checkpoints/navier-1/navier-1-beta-1512/ckpt-60000
export DATASET_NAME=dresscode
export DATAPATH=/hosjiu/data/DressCode
export OUTPUT_DIR=results/eval
export NUM_WORKERS=4
export BATCH_SIZE=8

python inference.py \
--model_path=$MODEL_PATH \
--dataset_name=$DATASET_NAME \
--dresscode_datapath=$DATAPATH \
--output_dir=$OUTPUT_DIR \
--num_workers=$NUM_WORKERS \
--batch_size=$BATCH_SIZE \
--eval