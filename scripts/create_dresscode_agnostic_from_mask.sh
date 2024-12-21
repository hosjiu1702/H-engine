#!/bin/bash
export base_path=$1

python -u src/dataset/utils.py create_agnostic_from_mask_for_dresscode $base_path
