#!/bin/bash
#SBATCH --job-name=xettat-eval
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/evaluate.sbatch RUN_ID [SPLIT] [DATA_DIR]
#   SPLIT: val | test | real (default val); DATA_DIR default data/processed (Wikipedia).
#   With another DATA_DIR (data/hplt, data/subs) results go to runs/RUN_ID/final_SPLIT_<dirname>.json
# Use 'test' only once per model, for the final numbers.
set -euo pipefail
RUN_ID=${1:?usage: sbatch slurm/evaluate.sbatch RUN_ID [SPLIT]}
SPLIT=${2:-val}
DATA=${3:-data/processed}
TAG=""; [ "$DATA" != "data/processed" ] && TAG=$(basename "$DATA")
cd "$SLURM_SUBMIT_DIR"
CKPT_ROOT=${XETTAT_CKPT_ROOT:-/data/teams/team14/checkpoints}
python -m xettat.evaluate --ckpt "$CKPT_ROOT/$RUN_ID/best.pt" --data "$DATA" --split "$SPLIT" --out "runs/$RUN_ID" --tag "$TAG"
bash slurm/next_stage.sh
