#!/bin/bash
#SBATCH --job-name=fastlenet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=60-24:00:00
#SBATCH --output=logs/outputlenetcnn.log
#SBATCH --error=logs/errorlenetcnn.log
#SBATCH --mem=124G
# Load necessary modules or activate virtual environment
module load cuda-toolkit/11.6.2
module load python/3.8.6

torchrun --nproc_per_node=2 lenetcnn.py -p model_snapshots/lenetcnn -e 50 -s 2
