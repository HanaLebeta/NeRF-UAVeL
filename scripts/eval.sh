#!/bin/bash
# eval.sh - Evaluate NeRF-UAVeL pretrained models
# Usage: bash scripts/eval.sh [front3d|scannet] [checkpoint_path] [gpu_id]

set -e

DATASET=${1:-"front3d"}
CHECKPOINT=${2:-""}
GPU=${3:-0}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: bash scripts/eval.sh [front3d|scannet] [checkpoint_path] [gpu_id]"
    echo "Example: bash scripts/eval.sh front3d results/front3d_full/model_best.pt 0"
    exit 1
fi

if [ "$DATASET" == "front3d" ]; then
    DATA_ROOT=data/front3d_rpn_data
    SPLIT=${DATA_ROOT}/3dfront_split.npz
    SAVE_PATH=results/front3d_eval
elif [ "$DATASET" == "scannet" ]; then
    DATA_ROOT=data/scannet_rpn_data
    SPLIT=${DATA_ROOT}/scannet_split.npz
    SAVE_PATH=results/scannet_eval
else
    echo "Unknown dataset: $DATASET. Use 'front3d' or 'scannet'."
    exit 1
fi

echo "============================================"
echo "Evaluating NeRF-UAVeL on ${DATASET}"
echo "Checkpoint: ${CHECKPOINT}"
echo "GPU: ${GPU}"
echo "============================================"

python3 -u nerf_rpn/run_fcos.py \
    --mode eval \
    --dataset ${DATASET} \
    --resolution 160 \
    --backbone_type vgg_EF \
    --features_path ${DATA_ROOT}/features \
    --boxes_path ${DATA_ROOT}/obb \
    --dataset_split ${SPLIT} \
    --save_path ${SAVE_PATH} \
    --checkpoint ${CHECKPOINT} \
    --norm_reg_targets \
    --centerness_on_reg \
    --rotated_bbox \
    --normalize_density \
    --nms_thresh 0.3 \
    --output_proposals \
    --save_level_index \
    --batch_size 2 \
    --gpus ${GPU}

echo "============================================"
echo "Evaluation complete. Results saved to ${SAVE_PATH}"
echo "============================================"
