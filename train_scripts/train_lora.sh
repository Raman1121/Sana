#! /bin/bash

export MODEL_NAME="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
export OUTPUT_DIR="trained-sana-lora"
export BATCH_SIZE=16
export LR=1e-4
export MAX_TRAIN_STEPS=9000

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
  # --push_to_hub
