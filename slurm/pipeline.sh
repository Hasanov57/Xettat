#!/bin/bash
# Run every stage overnight with ONE command:   bash slurm/pipeline.sh
# Stages run one after another; each finished stage submits the next. A failed stage stops the chain
# (see logs/). Check progress:  cat .pipeline_queue ; squeue --me ; ls -t logs | head
# Skip data preparation if it is already done:   bash slurm/pipeline.sh --skip-data
# Data job already running (e.g. job 25)? Start the rest when it succeeds:   bash slurm/pipeline.sh --after 25
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
cat > .pipeline_queue <<'Q'
slurm/train.sbatch configs/bilstm.yaml bilstm_v1
slurm/evaluate.sbatch bilstm_v1 val
slurm/train.sbatch configs/transformer.yaml transformer_v1
slurm/evaluate.sbatch transformer_v1 val
Q
if [ "${1:-}" = "--skip-data" ]; then
  bash slurm/next_stage.sh
elif [ "${1:-}" = "--after" ]; then
  JOB=${2:?usage: bash slurm/pipeline.sh --after JOBID}
  FIRST=$(head -n 1 .pipeline_queue)
  tail -n +2 .pipeline_queue > .pipeline_queue.tmp && mv .pipeline_queue.tmp .pipeline_queue
  # shellcheck disable=SC2086
  sbatch --dependency=afterok:"$JOB" --kill-on-invalid-dep=yes $FIRST
else
  sbatch slurm/prepare_data.sbatch
fi
echo "[pipeline] started. Remaining stages:"; cat .pipeline_queue
