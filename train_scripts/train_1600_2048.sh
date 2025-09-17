#/bin/bash
set -e

RESOLUTION=2048
SIZE=1600M
work_dir=output/Sana_${SIZE}_${RESOLUTION}
np=6


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="configs/sana_config/${RESOLUTION}ms/Sana_${SIZE}_img${RESOLUTION}_bf16.yaml"
    # config="configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml"      FSDP config file
    echo "Only support .yaml files, but get $1. Set to --config_path=$config"
fi

TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=15432 \
        train_scripts/train.py \
        --config_path=$config \
        --data.type=SanaImgDataset \
        --model.load_from="hf://Efficient-Large-Model/Sana_1600M_2Kpx_BF16/checkpoints/Sana_1600M_2Kpx_BF16.pth" \
        --model.multi_scale=false \
        --train.train_batch_size=32 \
        --work_dir=$work_dir \
        --tracker_project_name="PulmoGen-Training" \
        --name=PulmoGen_${SIZE}_${RESOLUTION} \
        --debug=false \
        "$@"

        # --resume_from=latest \
