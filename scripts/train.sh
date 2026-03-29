#!/bin/bash
# train.sh - Train NeRF-UAVeL
# Usage: bash scripts/train.sh [front3d|scannet] [gpu_ids]

set -e

DATASET=${1:-"front3d"}
GPUS=${2:-0}

if [ "$DATASET" == "front3d" ]; then
    DATA_ROOT=data/front3d_rpn_data
    SPLIT=${DATA_ROOT}/3dfront_split.npz
    SAVE_PATH=results/front3d_full
    NUM_EPOCHS=1250
elif [ "$DATASET" == "scannet" ]; then
    DATA_ROOT=data/scannet_rpn_data
    SPLIT=${DATA_ROOT}/scannet_split.npz
    SAVE_PATH=results/scannet_full
    NUM_EPOCHS=1200
else
    echo "Unknown dataset: $DATASET. Use 'front3d' or 'scannet'."
    exit 1
fi

echo "============================================"
echo "Training NeRF-UAVeL on ${DATASET}"
echo "Epochs: ${NUM_EPOCHS}"
echo "GPUs: ${GPUS}"
echo "============================================"

python3 -u nerf_rpn/run_fcos.py \
    --mode train \
    --dataset ${DATASET} \
    --resolution 160 \
    --backbone_type vgg_EF \
    --features_path ${DATA_ROOT}/features \
    --boxes_path ${DATA_ROOT}/obb \
    --dataset_split ${SPLIT} \
    --save_path ${SAVE_PATH} \
    --num_epochs ${NUM_EPOCHS} \
    --lr 3e-4 \
    --weight_decay 1e-3 \
    --log_interval 10 \
    --eval_interval 10 \
    --norm_reg_targets \
    --centerness_on_reg \
    --center_sampling_radius 1.5 \
    --iou_loss_type iou \
    --rotated_bbox \
    --log_to_file \
    --normalize_density \
    --nms_thresh 0.3 \
    --batch_size 2 \
    --gpus ${GPUS}

echo "============================================"
echo "Training complete. Model saved to ${SAVE_PATH}"
echo "============================================"
