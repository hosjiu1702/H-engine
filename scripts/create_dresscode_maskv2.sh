#!/bin/bash
BASE_PATH=datasets/dresscode
OVERWRITE=true
CATEGORY=dresses

python -u src/dataset/utils.py create_mask_v2_for_dresscode --base_path=$BASE_PATH/$CATEGORY/ --category=$CATEGORY --overwrite=$OVERWRITE