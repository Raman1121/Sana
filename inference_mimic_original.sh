#!/bin/bash

RESOLUTION=1024
SIZE=1600

CONFIG_FILE="/pvc/Sana/configs/sana_config/${RESOLUTION}ms/Sana_${SIZE}M_img${RESOLUTION}.yaml"
CKPT_PATH="/pvc/Sana/output/Llavarad_Captions/Sana_${SIZE}M_${RESOLUTION}/checkpoints/epoch_20_step_37101.pth"
# PROMPTS_TEXT_FILE="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_test_prompts.txt"
PROMPTS_TEXT_FILE="/pvc/CheXGenBench/MIMIC_Splits/ORIGINAL_ANNOTATIONS_TEST.txt"

python scripts/inference.py \
        --config=$CONFIG_FILE \
        --model_path=$CKPT_PATH \
        --txt_file=$PROMPTS_TEXT_FILE \