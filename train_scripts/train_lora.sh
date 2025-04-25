#! /bin/bash

export MODEL_NAME="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
export OUTPUT_DIR="trained-sana-lora"
export BATCH_SIZE=16
export LR=1e-4
export MAX_TRAIN_STEPS=100

TRAIN_CSV="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/LLAVARAD_ANNOTATIONS_TRAIN.csv"
TEST_CSV="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/LLAVARAD_ANNOTATIONS_TEST.csv"
IMG_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
IMG_COL="path"
CAPTION_COL="annotated_prompt"

accelerate launch --num_processes 2 --main_process_port 29500 \
  train_scripts/train_dreambooth_lora_sana.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --learning_rate=$LR \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --validation_epochs=5 \
  --seed="0" \
  --train_csv=$TRAIN_CSV \
  --test_csv=$TEST_CSV \
  --train_data_dir=$IMG_DIR \
  --image_column=$IMG_COL \
  --caption_column=$CAPTION_COL \
  --checkpoints_total_limit=1 --checkpointing_steps=1000 \
  # --push_to_hub
