export DATASET='vitonhd'
export DATAROOT='datasets/vitonhd'
export OUTPUT_DIR='results/emasc'
export MODEL_ID='stable-diffusion-v1-5/stable-diffusion-inpainting'
export CONFIG_FILE='configs/train_emasc.yaml'

accelerate launch --config_file=$CONFIG_FILE train_emasc.py --dataset=$DATASET --vitonhd_dataroot=$DATAROOT --output_dir=$OUTPUT_DIR --pretrained_model_name_or_path=$MODEL_ID