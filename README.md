<div align="center">
<h1>H-engine
</div>

**H-engine**, which stands for *Heatmob VTO Engine*, aims to build a full-fledge virtual try-on (VTO) engine that support for training, testing, evaluating,... and serving as a simple codebase to try conduct many research ideas as quick as possible in terms of implementation.

---
> [!WARNING]
> This codebase is highly developed so the everything could be broken accidently.

## TODO LIST
- [x] training code *(still under active development)*
- [ ] inference code
- [ ] evaluation
- [ ] Gradio demo

## Project Structure
```
vto-engine
|   README.md
|   inference.py
|   requirements.txt
|   setup.py
|   train.py
|   ...
|___assets
|___checkpoints
|___datasets # put your dataset here for training
|___experiment # for experiments that not directly relate to the core source
|___gradio_demo # for gradio demo
|___legacy # this aim to reproduce the old vto pipeline for comparison with new models
|___notebooks
|___scripts # convention scripts
|___src
|   |___dataset
|   |___models # core models (new vto models are implemented here)
|   |___pipelines # have no use for now
|   |___utils
|   |___preprocess # compute mask, human segmentation,...
```

## Installation
#### 1. Prerequistes
- Python >= 3.8 (only tested 3.8)
- Pytorch 2.4.1
- Diffusers 0.30.3

#### 2. Installation via pip (venv module)
```
python3.8 -m venv .venv
source .venv/bin/activate
pip install -U pip

pip install -r requirements.txt

pip install -e . (install this project as editable python module)
```

## Data Preparation
Download the **VITON-HD** dataset

```
./scripts/download.sh
```

Folder structure:
```
├── VITON-HD
|   ├── test_pairs.txt
|   ├── train_pairs.txt
│   ├── [train | test]
|   |   ├── image
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth-mask
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── image-parse-v3
│   │   │   ├── [000006_00.png | 000008_00.png | ...]
│   │   ├── openpose_img
│   │   │   ├── [000006_00_rendered.png | 000008_00_rendered.png | ...]
│   │   ├── openpose_json
│   │   │   ├── [000006_00_keypoints.json | 000008_00_keypoints.json | ...]
```

## Training
#### 1. Train on cpu
```
./scripts/train_cpu_only.sh
```

#### 2. Train on single gpu
```
./scripts/train_single_gpu.sh
```

> [!NOTE]
> Test and add support for multi-gpus soon! (almost done thanks to 🤗 Accelerate)

## Acknowledgements
This codebase and many other things get inspired or borrowed from awesome opensource projects below:

* [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
* [OOTDifusion](https://github.com/levihsu/OOTDiffusion)
* [IDM-VTON](https://github.com/yisol/IDM-VTON)
* [LaDI-VTON](https://github.com/miccunifi/ladi-vton)
* [CAT-DM](https://github.com/zengjianhao/CAT-DM)
* [StableVITON](https://github.com/rlawjdghek/StableVITON)
* [CatVTON](https://github.com/Zheng-Chong/CatVTON)
* [Diffusers](https://github.com/huggingface/diffusers)