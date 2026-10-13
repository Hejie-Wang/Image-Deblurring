#!/bin/bash
# DeepDeblur 训练启动脚本（服务器上运行）
# 用法:
#   单卡/DataParallel:  bash run_train.sh
#   3 卡 DDP:           bash run_train.sh ddp
set -e
cd "$(dirname "$0")"

DATA_ROOT=${DATA_ROOT:-~/datasets/GOPRO_Large}
EPOCHS=${EPOCHS:-100}

if [ "$1" = "ddp" ]; then
    torchrun --nproc_per_node=3 train.py \
        --data_root "$DATA_ROOT" \
        --save_dir experiment_ddp \
        --epochs "$EPOCHS" \
        --batch_size 24 \
        --num_workers 8
else
    python train.py \
        --data_root "$DATA_ROOT" \
        --save_dir experiment \
        --epochs "$EPOCHS" \
        --batch_size 24 \
        --num_workers 8
fi
