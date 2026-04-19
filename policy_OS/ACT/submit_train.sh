#!/bin/bash
#SBATCH --job-name=univtac_train
#SBATCH --partition=gpu-v100 
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

cd /scratch/zt1/project/ssb-prj/user/osaha/848M/UniVTAC/policy/ACT

export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

#bash train.sh insert_HDMI clean 100 0 0#
bash train.sh insert_HDMI clean 100 0 0 train_config_vision_all --wandb_mode offline
