#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=TDG-VAD
#SBATCH --partition=gpu

python test.py --config ./configs/test.yaml


