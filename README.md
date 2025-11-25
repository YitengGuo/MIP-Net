# RHCNet: Residual-Guided Hierarchical Calibration Network for Robust Underwater Object Detection

This repository contains the code (in PyTorch) 

## Introduction

Underwater object often suffer from severe visual degradation, including strong background noise induced by water impurities, blurred object boundaries caused by optical scattering, and small object suppression under complex environments, which collectively pose significant challenges to underwater object detection (UOD). To address these issues, we propose a Multi-Scale Inverted Pyramid Network (MIP-Net) tailored for underwater object detection. MIP-Net integrates two key components: the Local Adaptive Contrast module (LAC) and the Multi-Scale Inverted Feature Pyramid Network (MSIFPN). The LAC module enhances high-frequency details through layer-specific contrast calibration and dynamic feature modulation, improving discriminability against fuzzy and small objects. Meanwhile, MSIFPN employs a dual-pyramid structure with upright and inverted pyramids, coupled with a foreground background separation strategy, which effectively suppresses noise from water impurities while retaining salient features. Experiments on a blurred underwater dataset demonstrate that MIP-Net achieves an AP of 70.1\%, surpassing state-of-the-art methods. Furthermore, evaluations on a terrestrial dataset yield an AP of 45.6\%, confirming the strong generalization ability and robustness of MIP-Net across diverse environments.
![pipeline](./model.png)

## Dependencies

- Python == 3.7.11
- PyTorch == 1.10.1
- mmdetection == 2.22.0
- mmcv == 1.4.0
- numpy == 1.21.2

## Installation

The basic installation follows with [mmdetection](https://github.com/mousecpn/mmdetection/blob/master/docs/get_started.md). It is recommended to use manual installation. 

## Datasets

**DUO**: https://github.com/chongweiliu/DUO

**UTDAC2020**: https://drive.google.com/file/d/1avyB-ht3VxNERHpAwNTuBRFOxiXDMczI/view?usp=sharing


Other underwater datasets: https://github.com/mousecpn/Collection-of-Underwater-Object-Detection-Dataset

After downloading all datasets, create udmdet document.

```
$ cd data
$ mkdir udmdet
```

It is recommended to symlink the dataset root to `$data`.

```
udmdet
├── data
│   ├── DUO
│   │   ├── annotaions
│   │   ├── train2017
│   │   ├── test2017
```


## Train

```
$ python tools/train.py configs/mipnet/mipnet_tood_r50_fpn_anchor_based_2x_duoc.py
```

## Test

```
$ python tools/test.py configs/mipnet/mipnet_tood_r50_fpn_anchor_based_2x_duo.py <path/to/checkpoints> --eval bbox
```




## Results

![pipeline](./result.png)



## Acknowledgement

Thanks MMDetection team for the wonderful open source project!



