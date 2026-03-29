# Pretrained Checkpoints

**Coming Soon**

Pretrained model checkpoints for NeRF-UAVeL will be released here.

## Expected Checkpoints

| Model | Dataset | Backbone | Resolution | File |
|-------|---------|----------|------------|------|
| NeRF-UAVeL | 3D-FRONT | vgg_EF | 160 | `front3d_best.pth` |
| NeRF-UAVeL | ScanNet | vgg_EF | 160 | `scannet_best.pth` |

## Usage

Once available, download the checkpoint files into this directory and run evaluation:

```bash
# Evaluate on 3D-FRONT
bash scripts/eval.sh front3d checkpoints/front3d_best.pth

# Evaluate on ScanNet
bash scripts/eval.sh scannet checkpoints/scannet_best.pth
```
