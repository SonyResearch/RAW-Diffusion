# RAW Object Detection

## Installation

Please install the conda environment as described in [README](../README.md).
Afterward, mmdetection can be installed by following the official installation steps: [https://mmdetection.readthedocs.io/en/3.x/get_started.html](https://mmdetection.readthedocs.io/en/3.x/get_started.html).

```bash
cd object_detection
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"

pip install -v -e .
```

## Experiments

Object detection experiments are performed on NOD Nikon and Sony.

### NOD Nikon

#### RGB
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_nikon_rgb_pIN.py --cfg-options randomness.seed=0
```

#### RAW
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_nikon_raw3c_pIN_csraw.py --cfg-options train_dataloader.dataset.f=1.0 randomness.seed=0
```

#### RAW + Cityscapes-RAW (RAW-Diffusion)
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_nikon_raw3c_pIN_csraw.py --cfg-options train_dataloader.dataset.f=0.05 randomness.seed=0
```

#### RAW + BDD100K-RAW (RAW-Diffusion)
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_nikon_raw3c_pIN_bddraw.py --cfg-options train_dataloader.dataset.f=0.05 randomness.seed=0
```

### NOD Sony

#### RGB
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_sony_rgb_pIN.py --cfg-options randomness.seed=0
```

#### RAW
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_sony_raw3c_pIN_csraw.py --cfg-options train_dataloader.dataset.f=1.0 randomness.seed=0
```


#### RAW + Cityscapes-RAW (RAW-Diffusion)
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_sony_raw3c_pIN_csraw.py --cfg-options train_dataloader.dataset.f=0.05 randomness.seed=0
```

#### RAW + BDD100K-RAW (RAW-Diffusion)
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_sony_raw3c_pIN_bddraw.py --cfg-options train_dataloader.dataset.f=0.05 randomness.seed=0
```

## Zero-Shot Experiments

### NOD Nikon

#### Cityscapes-RAW (RAW-Diffusion)
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_nikon_raw3c_pIN_csraw.py --cfg-options train_dataloader.dataset.f=0.0 randomness.seed=0
```

#### BDD100K-RAW (RAW-Diffusion)
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_nikon_raw3c_pIN_bddraw.py --cfg-options train_dataloader.dataset.f=0.0 randomness.seed=0
```


### NOD Sony

#### Cityscapes-RAW (RAW-Diffusion)
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_sony_raw3c_pIN_csraw.py --cfg-options train_dataloader.dataset.f=0.0 randomness.seed=0
```

#### BDD100K-RAW (RAW-Diffusion)
```bash
python tools/train.py raw/configs/faster-rcnn_r50_fpn_nod_h416_sony_raw3c_pIN_bddraw.py --cfg-options train_dataloader.dataset.f=0.0 randomness.seed=0
```
