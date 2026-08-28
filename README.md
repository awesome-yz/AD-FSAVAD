# Adversarial Diffusion for Few-Shot Scene Adaptive Video Anomaly Detection

This repository contains the official PyTorch implementation for **Adversarial Diffusion for Few-Shot Scene-Adaptive Video Anomaly Detection**. 

Our framework introduces a novel pipeline that integrates a **Denoising Diffusion model** with a future-frame prediction architecture inside a **Generative Adversarial Network (GAN)** meta-learning setup. By optimizing one-step high-fidelity future-frame synthesis and leveraging perceptual regularization metrics (MS-SSIM/PSNR), our model achieves rapid, high-performance scene adaptation under minimal episodic training iterations.

---
<p align="center">
  <img src="images/architecture.png" alt="System Architecture" width="700">
</p>

---
## 1. Data Preparation:

Download dataset and update **train.yaml** config file:

train_frame_path: [provide path to folder containing training frames]

test_frame_path: [provide path to folder containing testing frames]

gt_folder: [provide path to folder containg grountruth masks]

### Datasets:
ShanghaiTech: https://svip-lab.github.io/dataset/campus_dataset.html

CUHK Avenue: https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

UCSD Ped1&2: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm


## 2. Training

Update options and provide paths of the datasets in the config files (train.yaml) and run:

```
python train.py --config ./configs/train.yaml
```

## 3. Evaluation
Update **test.yaml** config file and run:

```
python test.py --config ./configs/test.yaml
```
---
## Acknowledgments
This repository adapts and builds upon code from the following projects:
* [Few-shot-Scene-adaptive-Anomaly-Detection] (https://github.com/yiweilu3/Few-shot-Scene-adaptive-Anomaly-Detection.git)
---
## Citation
If you find this repository useful, please cite our paper:

```bibtex
@article{yuz2024adversarial,
  title={Adversarial diffusion for few-shot scene-adaptive video anomaly detection},
  journal={Neurocomputing},
  volume={567},
  pages={115674},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.neucom.2024.115674}
}
```
