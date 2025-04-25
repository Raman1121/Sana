#/bin/bash
set -e

work_dir=output/debug
np=1


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="configs/sana_config/1024ms/Sana_1600M_img1024.yaml"
    # config="configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml"      FSDP config file
    echo "Only support .yaml files, but get $1. Set to --config_path=$config"
fi

TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=15432 \
        train_scripts/train.py \
        --config_path=$config \
        --data.data_dir="[/pvc/MIMIC_ARRANGED/Train]" \
        --data.type=SanaImgDataset \
        --model.load_from="hf://Efficient-Large-Model/Sana_1600M_512px/checkpoints/Sana_1600M_512px.pth" \
        --model.multi_scale=false \
        --train.train_batch_size=256 \
        --work_dir=$work_dir \
        --name=tmp \
        --report_to=tensorboard \
        --debug=false \
        "$@"

        # --resume_from=latest \
