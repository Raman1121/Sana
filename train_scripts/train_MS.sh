#/bin/bash
set -e

RESOLUTION=512
SIZE=600M
work_dir=output/PulmoGen_${SIZE}_${RESOLUTION}_MS
np=1


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="configs/sana_config/${RESOLUTION}ms/Sana_${SIZE}_img${RESOLUTION}.yaml"
    # config="configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml"      FSDP config file
    echo "Only support .yaml files, but get $1. Set to --config_path=$config"
fi

TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=15432 \
        train_scripts/train.py \
        --config_path=$config \
        --data.type=SanaWebDatasetMS \
        --data.data_dir="[/pvc/MIMIC_ARRANGED_MS.tar,/pvc/MIMIC_ARRANGED_LATERAL_MS.tar,/pvc/Chexpert/chexpertchestxrays-u20210408/TRAIN_ARRANGED_MS.tar,/pvc/Chexpert/chexpertchestxrays-u20210408/TRAIN_ARRANGED_LATERAL_Chexpert_MS.tar,/pvc/ReXGradient-160K/ReXGradient-160K/deid_png_uint8_MS.tar]" \
        --model.load_from="hf://Efficient-Large-Model/Sana_${SIZE}_${RESOLUTION}px/checkpoints/Sana_${SIZE}_${RESOLUTION}px_MultiLing.pth" \
        --model.multi_scale=true \
        --train.train_batch_size=128 \
        --work_dir=$work_dir \
        --name=tmp \
        --report_to=tensorboard \
        --debug=false \
        "$@"

        # --resume_from=latest \
