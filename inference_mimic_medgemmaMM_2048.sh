#!/bin/bash

RESOLUTION=2048
SIZE=1600

CONFIG_FILE="/pvc/Sana/configs/sana_config/2048ms/Sana_1600M_img2048_bf16.yaml"
CKPT_PATH="/pvc/Sana/output/Llavarad_Captions/Sana_1600M_2048/checkpoints/epoch_20_step_95000.pth"
# PROMPTS_TEXT_FILE="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_test_prompts.txt"
PROMPTS_TEXT_FILE="/pvc/CheXGenBench/MIMIC_Splits/MEDGEMMA_MM_ANNOTATIONS_TEST.txt"

python scripts/inference.py \
        --config=$CONFIG_FILE \
        --model_path=$CKPT_PATH \
        --txt_file=$PROMPTS_TEXT_FILE \