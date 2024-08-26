#!/bin/bash
#SBATCH --account=bsc01
#SBATCH --qos=acc_bsccase
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1

module purge
module load mkl intel python tensorflow

script=$1
line=$2

python3 "$script" "$line"