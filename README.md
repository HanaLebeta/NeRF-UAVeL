<div align="center">

# NeRF-UAVeL

### Unified attention-driven volumetric learning for robust NeRF-based 3D object detection

<p>
<a href="https://doi.org/10.1016/j.neucom.2026.134610">
<img alt="Paper" src="https://img.shields.io/badge/Paper-Neurocomputing-b31b1b?style=for-the-badge&logo=elsevier&logoColor=white">
</a>
<a href="https://doi.org/10.1016/j.neucom.2026.134610">
<img alt="DOI" src="https://img.shields.io/badge/DOI-10.1016%2Fj.neucom.2026.134610-004f9f?style=for-the-badge">
</a>
<a href="https://hanalebeta.github.io/NeRF-UAVeL/">
<img alt="Project Page" src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github-pages&logoColor=white">
</a>
<a href="https://github.com/HanaLebeta/NeRF-UAVeL">
<img alt="Code" src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
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

**[Hana L. Goshu](https://www.hanagoshu.com)<sup>1</sup>, [Tadesse G. Wakjira](https://www.ai4riselab.com)<sup>2</sup>, Meklit M. Atlaw<sup>3</sup>, Kin-Chung Chan<sup>1</sup>, Songjiang Lai<sup>1</sup>, Kin-Man Lam<sup>1</sup>**

<sup>1</sup>The Hong Kong Polytechnic University &nbsp; <sup>2</sup>Kennesaw State University &nbsp; <sup>3</sup>Xi'an Jiaotong University

*Published in [Neurocomputing](https://doi.org/10.1016/j.neucom.2026.134610), Volume 702 (2026), 134610*

</div>

---

<div align="center">
<img src="assets/demo_front3d.gif" width="95%">
<br>
<em>3D object detection results on 3D-FRONT with predicted bounding boxes.</em>
</div>

<div align="center">
<img src="assets/demo_scannet.gif" width="95%">
<br>
<em>3D object detection results on ScanNet with predicted bounding boxes.</em>
</div>

---

## Abstract

Three-dimensional (3D) object detection based on Neural Radiance Fields (NeRF) has emerged as a promising direction for reconstructing complex environments from posed RGB images. However, existing NeRF-based detectors often suffer from coarse feature encoding and limited attention to multi-scale volumetric structures, leading to inaccurate localization and poor generalization in real-world scenarios. To address these challenges, we propose **NeRF-UAVeL**, a unified attention-driven volumetric learning framework for 3D object detection that integrates four novel modules into a NeRF-derived 3D volumetric backbone, namely, Multi-dimensional Volumetric Attention Pooling (MVAP), Tri-Scale Asymmetric Convolutional Aggregation (TACA), Dual-Domain Attention Fusion (DDAF), and Volumetric Cross-Window Attention Fusion (V-CWAF). MVAP enhances spatial selectivity through adaptive attention-based pooling, TACA captures multi-scale volumetric features through asymmetric convolutional branches, DDAF applies channel and multi-scale spatial recalibration for refined feature emphasis, and V-CWAF applies windowed self-attention with dual-stage channel recalibration to boost high-level semantic encoding. Extensive experiments on the 3D-FRONT and ScanNet datasets demonstrate that NeRF-UAVeL outperforms both point cloud-based and multi-view-based methods. Specifically, it improves AP50 by 6.7 and R50 by 7.3 over the baseline on 3D-FRONT, and achieves a 6.9 improvement in AP50 and 2.9 in R50 on the ScanNet dataset. These results confirm the effectiveness of our attention-calibrated, multi-scale volumetric architecture in producing precise and robust 3D bounding box predictions across both synthetic and real-world scenes. The project page is available at https://hanalebeta.github.io/NeRF-UAVeL/.

## Main Results

Quantitative results on the 3D-FRONT and ScanNet datasets. The first block includes point-cloud-based methods, while the remaining entries are multi-view-based detection methods.

<table>
<thead>
<tr>
<th rowspan="2" align="left">Method</th>
<th colspan="4" align="center">3D-FRONT</th>
<th colspan="4" align="center">ScanNet</th>
</tr>
<tr>
<th align="center">R<sub>25</sub></th>
<th align="center">R<sub>50</sub></th>
<th align="center">AP<sub>25</sub></th>
<th align="center">AP<sub>50</sub></th>
<th align="center">R<sub>25</sub></th>
<th align="center">R<sub>50</sub></th>
<th align="center">AP<sub>25</sub></th>
<th align="center">AP<sub>50</sub></th>
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
<td align="center"><strong>66.8</strong></td>
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
<tr><td colspan="9"></td></tr>
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
<td align="left">NeRF-MAE*</td>
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
<tr><td colspan="9"></td></tr>
<tr>
<td align="left"><strong>NeRF-UAVeL (Ours)</strong></td>
<td align="center"><strong>98.5</strong></td>
<td align="center"><strong>77.2</strong></td>
<td align="center"><strong>87.3</strong></td>
<td align="center"><strong>66.6</strong></td>
<td align="center"><strong>92.6</strong></td>
<td align="center"><strong>45.8</strong></td>
<td align="center">60.1</td>
<td align="center"><strong>25.3</strong></td>
</tr>
</tbody>
</table>

<sup>*</sup> NeRF-MAE results include pretraining on 3D-FRONT and two additional external datasets (Hypersim and HM3D).

## Code Availability

The paper is published. The source code, pretrained models, and training scripts are being prepared for release and will be added to this repository.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{nerf_uavel2026,
    title     = {NeRF-UAVeL: Unified attention-driven volumetric learning for robust NeRF-based 3D object detection},
    author    = {Goshu, Hana L. and Wakjira, Tadesse G. and Atlaw, Meklit M. and Chan, Kin-Chung and Lai, Songjiang and Lam, Kin-Man},
    journal   = {Neurocomputing},
    volume    = {702},
    pages     = {134610},
    year      = {2026},
    issn      = {0925-2312},
    doi       = {10.1016/j.neucom.2026.134610}
}
```

## Acknowledgements

This project builds upon the [NeRF-RPN](https://github.com/lyclyc52/NeRF_RPN) codebase.

## License

This project is released under the [Apache 2.0 License](LICENSE).
