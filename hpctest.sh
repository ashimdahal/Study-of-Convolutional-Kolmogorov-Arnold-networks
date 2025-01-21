#!/bin/bash
#SBATCH --job-name=test_alexnet_kan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=60-24:00:00
#SBATCH --output=logs/outputtestalexnetkanvalexnetcnn.log
#SBATCH --error=logs/errortestalexkan.log
#SBATCH --mem=124G
# Load necessary modules or activate virtual environment
module load cuda-toolkit/11.6.2
module load python/3.8.6

torchrun --nproc_per_node=1 scripts/test_models.py 
