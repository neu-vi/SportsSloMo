# SportsSloMo 	

[![ArXiv](https://img.shields.io/badge/ArXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2212.03814)
[![GitHub Stars](https://img.shields.io/github/stars/neu-vi/SportsSloMo?style=social)](https://github.com/neu-vi/SportsSloMo)
[![Demo views](https://img.shields.io/youtube/views/FNMqoETB1sU)](https://www.youtube.com/watch?v=FNMqoETB1sU)

**This is the official repository of SportsSloMo**

\[[ArXiv Paper](https://arxiv.org/abs/2212.03814)\] \[[Demo(YouTube)](https://www.youtube.com/watch?v=FNMqoETB1sU)\] \[[Project Page](https://neu-vi.github.io/SportsSlomo/)\]

We provide the video clips along with the code and trained weights for SportsSloMo. If you find this repository useful, please consider citing our [paper](https://arxiv.org/abs/2308.16876).

### Examples of the sports scenes in SportsSloMo 
<center><img src="figures/sportsquali.png" width="100%"></center>
We propose a challenging dataset consisting of high-resolution (≥720p) sports videos with careful curation, tailored for human-centric scenarios. Our dataset covers 22 various sports categories with different content and motion patterns, including hockey, base- ball, skating, basketball, running, volleyball, etc.

## Table of Contents
1. [SportsSloMo Dataset](#SportsSloMoDataset)
1. [Requirements](#Requirements)
1. [Reference](#Reference)
1. [Contact](#Contact)

## SportsSloMo Dataset
#### Dataset intro
Our dataset is targeted for human-centric scenarios covering various sports, and we provide over 8k long video clips that can be used for video frame interpolation, video super-resolution, group activity recognition and dynamic view synthesis.

Video sequences of SportsSloMo dataset can be downloaded at the [link](https://vil.khoury.northeastern.edu/files/sportsslomo_video_seqs.zip).

Our dataset has following directory format:

```
├──── SportsSloMo_frames/
    ├──── clip_0000/
       ├──── frame_0000.png
             ├──── ...
    ├──── clip_0001/
       ├──── frame_0000.png
             ├──── ...
    ├──── ...
```

#### Preparation
```
mkdir SportsSloMo_video
cd SportsSloMo_video
## Download sportsslomo_video_seqs.zip to this directory
unzip sportsslomo_video_seqs.zip

## Extract frames for each clip
cd ./SportsSloMo
python scripts/extract_frames.py
```

## Requirements

In this repo, we provide the environment setting and sample code needed for plugging in auxiliary losses into the [EBME](https://github.com/srcn-ivl/EBME) model, and other VFI models are also compatibale unless specific packages need to be installed.

Our code is implemented using PyTorch 1.12.1 and Cuda 11.3, and we use CuPy package for forward warping operation (check [softmax-splatting](https://github.com/sniklaus/softmax-splatting) for details).

#### Installation

Create a conda environment and install dependencies:

```
git clone https://github.com/neu-vi/SportsSloMo.git
cd SportsSloMo

conda create --name SportsSloMo python=3.8
conda activate SportsSloMo

# Install the according versions of torch and torchvision
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install the according version of CuPy
pip install cupy_cuda113==10.6.0

# Install required packages 
pip install -U opencv-python
```

## Human-aware Loss Terms

TO improve the existing video frame interpolation models on our human-centric SportsSloMo benchmark, we introduce two loss terms considering the human-aware priors. We propose loss terms based on human segmentation in the panoptic setting and human keypoints estimation as extra supervision for intermediate frame synthesis.

### Pipeline overview
<center><img src="figures/pipeline.png" width="100%"></center>

### Data preparation
We provide scripts and instructions to acquire ground-truth detected keypoints and segmentation masks for our human-aware loss terms.

#### Keypoint data
```
python scripts/...
```

#### Segmentation data
```
python scripts/...
```


<!-- **Reference**:   -->
## Reference
> Jiaben Chen and Huaizu Jiang "SportsSloMo: A New Benchmark and Baselines for Human-centric Video Frame Interpolation"
> 
**BibTeX**
```bibtex
@article{chen2023sportsslomo,
  title={SportsSloMo: A New Benchmark and Baselines for Human-centric Video Frame Interpolation},
  author={Chen, Jiaben and Jiang, Huaizu},
  journal={arXiv preprint arXiv:2308.16876},
  year={2023}
}
```

## Contact
If you have any question, please send an email to [[Jiaben Chen](https://jiabenchen.github.io)] - jic088@ucsd.edu

## License
The source codes and datasets can be freely used for research and education only. Any commercial use should get formal permission first.
