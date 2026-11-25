#!/bin/bash
#SBATCH --job-name=xettat-data
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# CPU-only: download Azerbaijani Wikipedia, build frozen splits, score baselines on val.
# If the cluster refuses jobs without a GPU, add:  #SBATCH --gres=gpu:1g.20gb:1
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
python -m xettat.data.prepare --source wiki --out data/processed --seed 42
python -m xettat.run_baselines --data data/processed --split val --out runs/baselines
