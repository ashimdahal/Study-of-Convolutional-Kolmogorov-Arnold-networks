#!/bin/bash
#SBATCH --job-name=fastalexnet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=60-24:00:00
#SBATCH --output=logs/outputfastalexnetkan.log
#SBATCH --error=logs/errorfastalexnetkan.log
#SBATCH --mem=124G
# Load necessary modules or activate virtual environment
module load cuda-toolkit/11.6.2
module load python/3.8.6

torchrun --nproc_per_node=2 alexnetfastkan.py -p model_snapshots/fastalexnet -e 90 -s 5 
