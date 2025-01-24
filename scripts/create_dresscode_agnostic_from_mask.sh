#!/bin/bash
export BASE_PATH=datasets/dresscode/dresses

python -u src/dataset/utils.py create_agnostic_from_mask_for_dresscode --base_path=$BASE_PATH
