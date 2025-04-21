#!/bin/bash

CONFIG_FILE="/pvc/Sana/configs/sana_config/512ms/Sana_1600M_img512.yaml"
CKPT_PATH="/pvc/Sana/output/debug/checkpoints/epoch_20_step_178057.pth"
PROMPTS_TEXT_FILE="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_test_prompts.txt"

python scripts/inference.py \
        --config=$CONFIG_FILE \
        --model_path=$CKPT_PATH \
        --txt_file=$PROMPTS_TEXT_FILE \