#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=DG-VAD
#SBATCH --partition=gpu

python train.py --config ./configs/train.yaml


