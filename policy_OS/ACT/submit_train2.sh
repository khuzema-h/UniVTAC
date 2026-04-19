#!/bin/bash
#SBATCH --job-name=univtac_train
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

cd /scratch/zt1/project/ssb-prj/user/osaha/848M/UniVTAC/policy/ACT || exit 1

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

VENV=/scratch/zt1/project/ssb-prj/user/osaha/848M/UniVTAC/policy/ACT/act_env
PY=$VENV/bin/python

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

echo "Using python: $PY"
$PY -c "import sys; print(sys.executable)"
$PY -c "import torch; print(torch.__file__)"
$PY -c "import typing_extensions; print(typing_extensions.__file__)"
$PY -c "import torch; print('torch cuda available =', torch.cuda.is_available())"

bash train.sh insert_HDMI clean 100 0 0 train_config_vision_all