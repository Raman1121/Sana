#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate sana

CONFIG_FILE="/raid/s2198939/Sana/configs/sana_config/512ms/Sana_600M_img512.yaml"
CKPT_PATH="/raid/s2198939/Sana/output/debug/checkpoints/epoch_20_step_178057.pth"
PROMPTS_TEXT_FILE="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_test_prompts.txt"

CUDA_VISIBLE_DEVICES=7

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python scripts/inference.py \
                        --config=$CONFIG_FILE \
                        --model_path=$CKPT_PATH \
                        --txt_file=$PROMPTS_TEXT_FILE \