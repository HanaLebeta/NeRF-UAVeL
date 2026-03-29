#!/bin/bash
# NeRF-UAVeL Data Download Script
# Downloads preprocessed NeRF features and bounding box annotations
# from Hugging Face: https://huggingface.co/datasets/lyclyc52/NeRF_RPN/tree/main
#
# Prerequisites:
#   pip install huggingface_hub
#   (optional) huggingface-cli login  -- for gated datasets
#
# Usage: bash scripts/download_data.sh [front3d|scannet|all]

set -e

DATASET=${1:-all}
DATA_DIR="data"

# Hugging Face repository
HF_REPO="lyclyc52/NeRF_RPN"
HF_BASE_URL="https://huggingface.co/datasets/${HF_REPO}/resolve/main"

# ------------------------------------------------------------------
# Helper: download a file if it does not already exist
# ------------------------------------------------------------------
download_file() {
    local url="$1"
    local dest="$2"

    if [ -f "$dest" ]; then
        echo "    [skip] $dest already exists."
        return
    fi

    echo "    [download] $dest"
    mkdir -p "$(dirname "$dest")"
    wget -q --show-progress -O "$dest" "$url"
}

# ------------------------------------------------------------------
# 3D-FRONT data
# ------------------------------------------------------------------
download_front3d() {
    echo "==> Downloading 3D-FRONT data..."
    local dest_dir="${DATA_DIR}/front3d_rpn_data"
    mkdir -p "${dest_dir}"

    # Download using huggingface-cli (handles large files and LFS)
    if command -v huggingface-cli &> /dev/null; then
        echo "    Using huggingface-cli..."
        huggingface-cli download "${HF_REPO}" \
            --repo-type dataset \
            --local-dir "${dest_dir}" \
            --include "front3d_rpn_data/*"
    else
        echo "    huggingface-cli not found. Falling back to wget..."
        echo "    (Install with: pip install huggingface_hub)"
        echo ""

        # Download split file
        download_file \
            "${HF_BASE_URL}/front3d_rpn_data/3dfront_split.npz" \
            "${dest_dir}/3dfront_split.npz"

        # Download features and bounding boxes
        # Note: For large directories, huggingface-cli is recommended.
        echo "    [info] For complete feature/obb downloads, please use huggingface-cli:"
        echo "           pip install huggingface_hub"
        echo "           huggingface-cli download ${HF_REPO} --repo-type dataset --local-dir ${DATA_DIR}"
    fi

    echo "==> 3D-FRONT data download complete."
    echo ""
}

# ------------------------------------------------------------------
# ScanNet data
# ------------------------------------------------------------------
download_scannet() {
    echo "==> Downloading ScanNet data..."
    local dest_dir="${DATA_DIR}/scannet_rpn_data"
    mkdir -p "${dest_dir}"

    # Download using huggingface-cli (handles large files and LFS)
    if command -v huggingface-cli &> /dev/null; then
        echo "    Using huggingface-cli..."
        huggingface-cli download "${HF_REPO}" \
            --repo-type dataset \
            --local-dir "${dest_dir}" \
            --include "scannet_rpn_data/*"
    else
        echo "    huggingface-cli not found. Falling back to wget..."
        echo "    (Install with: pip install huggingface_hub)"
        echo ""

        # Download split file
        download_file \
            "${HF_BASE_URL}/scannet_rpn_data/scannet_split.npz" \
            "${dest_dir}/scannet_split.npz"

        # Download features and bounding boxes
        echo "    [info] For complete feature/obb downloads, please use huggingface-cli:"
        echo "           pip install huggingface_hub"
        echo "           huggingface-cli download ${HF_REPO} --repo-type dataset --local-dir ${DATA_DIR}"
    fi

    echo "==> ScanNet data download complete."
    echo ""
}

# ------------------------------------------------------------------
# Run requested downloads
# ------------------------------------------------------------------
echo "============================================"
echo " NeRF-UAVeL Data Download"
echo " Source: https://huggingface.co/datasets/${HF_REPO}"
echo "============================================"
echo ""

case "$DATASET" in
    front3d)
        download_front3d
        ;;
    scannet)
        download_scannet
        ;;
    all)
        download_front3d
        download_scannet
        ;;
    *)
        echo "Usage: bash scripts/download_data.sh [front3d|scannet|all]"
        exit 1
        ;;
esac

echo "Done. Data saved to: ${DATA_DIR}/"
