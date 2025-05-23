#!/bin/bash
#SBATCH --job-name=fastlenet_ablate          # Slurm job name
#SBATCH --partition=gpu                      # GPU partition on your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                         # 1 GPU is enough for MNIST sweep
#SBATCH --time=3-00:00:00                    # 3 days wall-time (plenty)
#SBATCH --mem=32G                            # RAM – MNIST fits easily
#SBATCH --output=logs/ablate_%j.out          # STDOUT  (%j = job-ID)
#SBATCH --error=logs/ablate_%j.err           # STDERR

# ── 1. Load modules / activate virtual-env ────────────────────────────────
module load cuda-toolkit/11.6.2
module load python/3.8.6
# If you use a venv or conda, activate it here
# source ~/envs/kan/bin/activate

# ── 2. Optional: make sure the logs folder exists ────────────────────────
mkdir -p logs
mkdir -p mnist_ablate_results

# ── 3. Launch the sweep (single-GPU) ─────────────────────────────────────
python train_mnist_ablate.py \
       --epochs 5 \                    # 5 epochs per variant (edit if needed)
       --gpus 1 \                      # internal flag – keep at 1
       --outdir mnist_ablate_results   # where CSV & plots are stored

# Slurm will write timing & resource-use in its epilogue automatically

