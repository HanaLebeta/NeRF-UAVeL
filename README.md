<div align="center">

<!-- PROJECT BANNER -->
<!-- Replace with your project banner image -->
<!-- <img src="assets/banner.png" width="100%"> -->

# NeRF-UAVeL

### Unified Attention-driven Volumetric Learning for Enhanced NeRF-based 3D Object Detection

<p>
<a href="#">
<img alt="Paper" src="https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white">
</a>
<a href="#">
<img alt="Project Page" src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github-pages&logoColor=white">
</a>
<a href="#">
<img alt="Python 3.8+" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
</a>
<a href="#">
<img alt="PyTorch 1.12+" src="https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
</a>
<a href="#">
<img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge&logo=apache&logoColor=white">
</a>
</p>

**[Authors]**

*Neurocomputing (Under Review)*

</div>

---

<div align="center">
<img src="assets/teaser.png" width="95%">
<br>
<em>NeRF-UAVeL achieves state-of-the-art 3D object detection from Neural Radiance Fields, with improvements of <strong>+6.7% AP50</strong> on 3D-FRONT and <strong>+6.9% AP50</strong> on ScanNet over the NeRF-RPN baseline.</em>
</div>

---

## Abstract

This paper presents **NeRF-UAVeL**, a unified attention-driven volumetric learning framework for 3D object detection from Neural Radiance Fields. The framework integrates four complementary modules into a volumetric backbone to enhance multi-scale feature representation and spatial context modeling:

1. **Multi-dimensional Volumetric Attention Pooling (MVAP)** -- decomposes volumetric attention into axis-specific components to capture directional feature asymmetry.
2. **Tri-Scale Asymmetric Convolutional Aggregation (TACA)** -- employs spatial-first, depth-second factorized convolutions for multi-scale feature extraction.
3. **Dual-Domain Attention Fusion (DDAF)** -- jointly recalibrates channel and spatial features through ECA-based attention and multi-rate dilated convolutions.
4. **Volumetric Cross-Window Attention Fusion (V-CWAF)** -- applies localized cross-window multi-head attention for global context modeling.

Extensive experiments on the **3D-FRONT** and **ScanNet** datasets demonstrate that NeRF-UAVeL achieves state-of-the-art performance across all metrics.

## Architecture

<div align="center">
<img src="assets/architecture.png" width="95%">
<br>
<em>Overview of the NeRF-UAVeL architecture.</em>
</div>

<br>

NeRF-UAVeL builds upon a VGG-19-inspired 3D volumetric backbone with a Feature Pyramid Network (FPN) and an FCOS-style detection head. The four proposed modules are strategically integrated at different stages of the backbone:

| Stage | Module | Role |
|:------|:-------|:-----|
| C2 -- C4 | **DDAF** | Channel + spatial attention recalibration |
| C5 | **V-CWAF** | Cross-window multi-head attention for global context |
| All pooling layers | **MVAP** | Replaces MaxPool3d with axis-decomposed attention pooling |
| C5 (last conv) | **TACA** | Multi-scale asymmetric convolution |

## Main Results

### Comparison with State-of-the-Art Methods

<table>
<thead>
<tr>
<th rowspan="2" align="left">Method</th>
<th colspan="4" align="center">3D-FRONT</th>
<th colspan="4" align="center">ScanNet</th>
</tr>
<tr>
<th align="center">R25</th>
<th align="center">R50</th>
<th align="center">AP25</th>
<th align="center">AP50</th>
<th align="center">R25</th>
<th align="center">R50</th>
<th align="center">AP25</th>
<th align="center">AP50</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left">VoteNet</td>
<td align="center">81.5</td>
<td align="center">61.6</td>
<td align="center">73.0</td>
<td align="center">49.6</td>
<td align="center">78.5</td>
<td align="center">34.2</td>
<td align="center">66.8</td>
<td align="center">18.2</td>
</tr>
<tr>
<td align="left">GroupFree</td>
<td align="center">84.9</td>
<td align="center">63.7</td>
<td align="center">72.1</td>
<td align="center">45.1</td>
<td align="center">75.2</td>
<td align="center">37.6</td>
<td align="center">60.1</td>
<td align="center">20.4</td>
</tr>
<tr>
<td align="left">FCAF3D</td>
<td align="center">89.1</td>
<td align="center">56.9</td>
<td align="center">73.1</td>
<td align="center">35.2</td>
<td align="center">90.2</td>
<td align="center">42.4</td>
<td align="center">63.7</td>
<td align="center">18.5</td>
</tr>
<tr>
<td align="left">ImVoxNet</td>
<td align="center">88.3</td>
<td align="center">71.5</td>
<td align="center">86.1</td>
<td align="center">66.4</td>
<td align="center">51.7</td>
<td align="center">20.2</td>
<td align="center">37.3</td>
<td align="center">9.8</td>
</tr>
<tr>
<td align="left">NeRF-MAE</td>
<td align="center">97.2</td>
<td align="center">74.5</td>
<td align="center">85.3</td>
<td align="center">63.0</td>
<td align="center">92.0</td>
<td align="center">39.5</td>
<td align="center">57.1</td>
<td align="center">17.0</td>
</tr>
<tr>
<td align="left">NeRF-RPN</td>
<td align="center">96.3</td>
<td align="center">69.9</td>
<td align="center">85.2</td>
<td align="center">59.9</td>
<td align="center">89.2</td>
<td align="center">42.9</td>
<td align="center">55.5</td>
<td align="center">18.4</td>
</tr>
<tr>
<td align="left"><strong>NeRF-UAVeL (Ours)</strong></td>
<td align="center"><strong>98.5</strong></td>
<td align="center"><strong>77.2</strong></td>
<td align="center"><strong>87.3</strong></td>
<td align="center"><strong>66.6</strong></td>
<td align="center"><strong>92.6</strong></td>
<td align="center"><strong>45.8</strong></td>
<td align="center"><strong>60.1</strong></td>
<td align="center"><strong>25.3</strong></td>
</tr>
</tbody>
</table>

## Installation

### Prerequisites

- Linux (tested on Ubuntu 20.04 / 22.04 / 24.04)
- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.3 (tested with CUDA 11.3, 11.8, 12.1, 13.0)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/<username>/NeRF-UAVeL.git
cd NeRF-UAVeL

# Create conda environment
conda create -n nerf_uavel python=3.10 -y
conda activate nerf_uavel

# Install PyTorch (adjust CUDA version to match your system)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# For CUDA 12.1:
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt

# Build CUDA extensions
cd nerf_rpn/model/rotated_iou/cuda_op
python setup.py install
cd ../../../..

cd nerf_rpn/model/rotated_align
python setup.py install
cd ../../..
```

## Data Preparation

We use the pre-extracted NeRF features provided by [NeRF-RPN](https://github.com/lyclyc52/NeRF_RPN). Download the data from:

> [https://huggingface.co/datasets/lyclyc52/NeRF_RPN/tree/main](https://huggingface.co/datasets/lyclyc52/NeRF_RPN/tree/main)

Organize the downloaded data as follows:

```
NeRF-UAVeL/
|-- data/
|   |-- front3d_rpn_data/
|   |   |-- features/          # Pre-extracted NeRF density features
|   |   |-- obb/               # Oriented bounding box annotations
|   |   |-- 3dfront_split.npz  # Train/val/test split
|   |-- scannet_rpn_data/
|   |   |-- features/
|   |   |-- obb/
|   |   |-- scannet_split.npz
```

## Training

### 3D-FRONT

```bash
python3 -u nerf_rpn/run_fcos.py \
    --mode train \
    --dataset front3d \
    --resolution 160 \
    --backbone_type vgg_EF \
    --features_path data/front3d_rpn_data/features \
    --boxes_path data/front3d_rpn_data/obb \
    --dataset_split data/front3d_rpn_data/3dfront_split.npz \
    --save_path results/front3d_full \
    --num_epochs 1250 \
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
    --gpus 0
```

### ScanNet

```bash
python3 -u nerf_rpn/run_fcos.py \
    --mode train \
    --dataset scannet \
    --resolution 160 \
    --backbone_type vgg_EF \
    --features_path data/scannet_rpn_data/features \
    --boxes_path data/scannet_rpn_data/obb \
    --dataset_split data/scannet_rpn_data/scannet_split.npz \
    --save_path results/scannet_full \
    --num_epochs 1200 \
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
    --gpus 0
```

## Evaluation

To evaluate a trained model, set `--mode` to `eval` and provide the checkpoint path:

```bash
# 3D-FRONT evaluation
python3 -u nerf_rpn/run_fcos.py \
    --mode eval \
    --dataset front3d \
    --resolution 160 \
    --backbone_type vgg_EF \
    --features_path data/front3d_rpn_data/features \
    --boxes_path data/front3d_rpn_data/obb \
    --dataset_split data/front3d_rpn_data/3dfront_split.npz \
    --save_path results/front3d_eval \
    --checkpoint results/front3d_full/model_best.pt \
    --norm_reg_targets \
    --centerness_on_reg \
    --rotated_bbox \
    --normalize_density \
    --nms_thresh 0.3 \
    --output_proposals \
    --batch_size 2 \
    --gpus 0

# ScanNet evaluation
python3 -u nerf_rpn/run_fcos.py \
    --mode eval \
    --dataset scannet \
    --resolution 160 \
    --backbone_type vgg_EF \
    --features_path data/scannet_rpn_data/features \
    --boxes_path data/scannet_rpn_data/obb \
    --dataset_split data/scannet_rpn_data/scannet_split.npz \
    --save_path results/scannet_eval \
    --checkpoint results/scannet_full/model_best.pt \
    --norm_reg_targets \
    --centerness_on_reg \
    --rotated_bbox \
    --normalize_density \
    --nms_thresh 0.3 \
    --output_proposals \
    --batch_size 2 \
    --gpus 0
```

## Pretrained Models

| Dataset | AP25 | AP50 | Download |
|:--------|:----:|:----:|:--------:|
| 3D-FRONT | 87.3 | 66.6 | Coming Soon |
| ScanNet | 60.2 | 25.3 | Coming Soon |

## Visualization

To visualize predicted 3D bounding boxes:

```bash
python3 -u nerf_rpn/run_fcos.py \
    --mode eval \
    --dataset front3d \
    --resolution 160 \
    --backbone_type vgg_EF \
    --features_path data/front3d_rpn_data/features \
    --boxes_path data/front3d_rpn_data/obb \
    --dataset_split data/front3d_rpn_data/3dfront_split.npz \
    --save_path results/front3d_vis \
    --rotated_bbox \
    --normalize_density \
    --nms_thresh 0.3 \
    --save_visualization \
    --gpus 0
```

Visualization outputs will be saved to the `--save_path` directory. You can inspect the predicted bounding boxes overlaid on the volumetric density grids using the included visualization utilities.

## Module-wise Ablation Study

<table>
<thead>
<tr>
<th rowspan="2" align="left">Configuration</th>
<th colspan="4" align="center">3D-FRONT</th>
<th colspan="4" align="center">ScanNet</th>
</tr>
<tr>
<th align="center">R25</th>
<th align="center">R50</th>
<th align="center">AP25</th>
<th align="center">AP50</th>
<th align="center">R25</th>
<th align="center">R50</th>
<th align="center">AP25</th>
<th align="center">AP50</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left">Baseline (NeRF-RPN)</td>
<td align="center">96.3</td>
<td align="center">69.9</td>
<td align="center">85.2</td>
<td align="center">59.9</td>
<td align="center">89.2</td>
<td align="center">42.9</td>
<td align="center">55.5</td>
<td align="center">18.4</td>
</tr>
<tr>
<td align="left">+ DDAF</td>
<td align="center">97.1</td>
<td align="center">72.8</td>
<td align="center">85.9</td>
<td align="center">62.0</td>
<td align="center">91.6</td>
<td align="center">41.4</td>
<td align="center">55.2</td>
<td align="center">19.2</td>
</tr>
<tr>
<td align="left">+ DDAF + TACA</td>
<td align="center">97.8</td>
<td align="center">73.5</td>
<td align="center">86.4</td>
<td align="center">61.1</td>
<td align="center">89.2</td>
<td align="center">42.4</td>
<td align="center">53.8</td>
<td align="center">22.1</td>
</tr>
<tr>
<td align="left">+ DDAF + MVAP</td>
<td align="center">97.1</td>
<td align="center">75.7</td>
<td align="center">86.7</td>
<td align="center">63.2</td>
<td align="center">90.1</td>
<td align="center">44.3</td>
<td align="center">57.5</td>
<td align="center">23.5</td>
</tr>
<tr>
<td align="left">+ DDAF + TACA + MVAP</td>
<td align="center">97.8</td>
<td align="center">76.5</td>
<td align="center">87.0</td>
<td align="center">64.4</td>
<td align="center">90.1</td>
<td align="center">43.8</td>
<td align="center">59.7</td>
<td align="center">24.6</td>
</tr>
<tr>
<td align="left"><strong>Full (NeRF-UAVeL)</strong></td>
<td align="center"><strong>98.5</strong></td>
<td align="center"><strong>77.2</strong></td>
<td align="center"><strong>87.3</strong></td>
<td align="center"><strong>66.6</strong></td>
<td align="center"><strong>92.6</strong></td>
<td align="center"><strong>45.8</strong></td>
<td align="center"><strong>60.1</strong></td>
<td align="center"><strong>25.3</strong></td>
</tr>
</tbody>
</table>

Each module contributes complementary improvements. MVAP provides the largest single-module gain on ScanNet (+5.1 AP50), while V-CWAF yields an additional +2.2 AP50 on 3D-FRONT when added to the DDAF+TACA+MVAP combination.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{nerf_uavel2025,
    title     = {NeRF-UAVeL: Unified Attention-driven Volumetric Learning for Enhanced NeRF-based 3D Object Detection},
    author    = {[Authors]},
    journal   = {Neurocomputing},
    year      = {2025},
    note      = {Under Review}
}
```

## Acknowledgements

This project builds upon the excellent [NeRF-RPN](https://github.com/lyclyc52/NeRF_RPN) codebase. We thank the authors for releasing their code and pre-extracted NeRF features. We also acknowledge the developers of the open-source libraries and datasets that made this work possible.

## License

This project is released under the [Apache 2.0 License](LICENSE).

---

<div align="center">
<em>If you have any questions, please open an issue or contact us.</em>
</div>
