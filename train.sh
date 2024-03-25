#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=dg-vad
#SBATCH --partition=gpu

python train.py --config ./configs/train.yaml


