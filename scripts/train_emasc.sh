# export TORCH_DISTRIBUTED_DEBUG=INFO
export DATASET='vitonhd'
export DATAROOT='datasets/vitonhd'
export OUTPUT_DIR='results/emasc'
export MODEL_ID='stable-diffusion-v1-5/stable-diffusion-inpainting'
export CONFIG_FILE='configs/train_emasc.yaml'
export BATCH_SIZE=4
export GRADIENT_ACCUMULATE_STEPS=2
export CHECKPOINTING_STEPS=10000

python -u -m accelerate.commands.launch --config_file=$CONFIG_FILE \
train_emasc.py \
--dataset=$DATASET \
--vitonhd_dataroot=$DATAROOT \
--output_dir=$OUTPUT_DIR \
--pretrained_model_name_or_path=$MODEL_ID \
--allow_tf32 \
--mixed_precision='bf16' \
--train_batch_size=$BATCH_SIZE \
--test_batch_size=$BATCH_SIZE \
--gradient_accumulation_steps=$GRADIENT_ACCUMULATE_STEPS \
--checkpointing_steps=$CHECKPOINTING_STEPS