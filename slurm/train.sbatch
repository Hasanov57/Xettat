#!/bin/bash
#SBATCH --job-name=xettat-train
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/train.sbatch configs/bilstm.yaml bilstm_v1
# Stops itself ~15 min before the 8 h limit, saves last.pt and resubmits; resumes automatically.
set -uo pipefail
CONFIG=${1:?usage: sbatch slurm/train.sbatch CONFIG RUN_ID}
RUN_ID=${2:?usage: sbatch slurm/train.sbatch CONFIG RUN_ID}
cd "$SLURM_SUBMIT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m xettat.train --config "$CONFIG" --run-id "$RUN_ID" --time-budget-min 465
code=$?
if [ "$code" -eq 99 ]; then
  echo "time budget reached -> resubmitting $RUN_ID"
  sbatch slurm/train.sbatch "$CONFIG" "$RUN_ID"
  exit 0
fi
if [ "$code" -eq 0 ]; then bash slurm/next_stage.sh; fi
exit "$code"
