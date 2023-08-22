# Unsupervised Non-rigid Point Cloud Registration Based on Point-wise Displacement Learning

## Description

This repository contains the code for our paper: Unsupervised Non-rigid Point Cloud Registration Based on Point-wise Displacement Learning

<div align="center">
<img src="https://github.com/djzgroup/Non-rigid-Registration/blob/main/images/Pipeline.png" width="70%" height="70%"><br><br>
</div>

## Prerequisities

Our models is trained and tested under:

- NVIDIAGPU+CUDA CuDNN

- PyTorch (torch==1.8.0)

- Python 3.6.9

- tqdm

- numpy

## Environment setup

```
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 torchaudio==0.8.0 -f 
```

## Dataset

The required dataset is in the ./data folder.

## Acknowledgment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

 [transformer](https://github.com/hyunwoongko/transformer) 

 [pointnet](https://github.com/charlesq34/pointnet) 