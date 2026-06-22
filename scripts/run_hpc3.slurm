#!/bin/bash
#SBATCH --job-name=qm9_gnn
#SBATCH --output=qm9_gnn_%j.out
#SBATCH --error=qm9_gnn_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Edit these lines for your HPC3 environment.
# module load python/3.11
# module load cuda

source ~/.bashrc
conda activate qm9-gnn

cd /path/to/qm9-property-prediction

python src/train.py \
  --subset -1 \
  --epochs 100 \
  --batch-size 128 \
  --hidden-dim 128 \
  --num-workers 4 \
  --device cuda
