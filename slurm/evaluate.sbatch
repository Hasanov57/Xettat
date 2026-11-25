#!/bin/bash
#SBATCH --job-name=xettat-eval
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/evaluate.sbatch RUN_ID [SPLIT]   (SPLIT: val | test | real; default val)
# Use 'test' only once per model, for the final numbers.
set -euo pipefail
RUN_ID=${1:?usage: sbatch slurm/evaluate.sbatch RUN_ID [SPLIT]}
SPLIT=${2:-val}
cd "$SLURM_SUBMIT_DIR"
CKPT_ROOT=${XETTAT_CKPT_ROOT:-/data/teams/team14/checkpoints}
python -m xettat.evaluate --ckpt "$CKPT_ROOT/$RUN_ID/best.pt" --data data/processed --split "$SPLIT" --out "runs/$RUN_ID"
