#!/bin/bash
#SBATCH --job-name=xettat-evvar
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/eval_variants.sbatch RUN_ID [SPLIT]
# Evaluates a checkpoint on every writing style (data/variants/<split>_<style>.tsv, built by
# python -m xettat.data.variants). Results: runs/RUN_ID/final_<split>_var_<style>.json
set -euo pipefail
RUN_ID=${1:?usage: sbatch slurm/eval_variants.sbatch RUN_ID [SPLIT]}
SPLIT=${2:-val}
cd "$SLURM_SUBMIT_DIR"
CKPT_ROOT=${XETTAT_CKPT_ROOT:-/data/teams/team14/checkpoints}
[ -f "data/variants/${SPLIT}_std.tsv" ] || python -m xettat.data.variants --data data/processed --split "$SPLIT" --out data/variants
for STYLE in std digraph a mixed english; do
  python -m xettat.evaluate --ckpt "$CKPT_ROOT/$RUN_ID/best.pt" --data data/processed --split "$SPLIT" \
    --tsv "data/variants/${SPLIT}_${STYLE}.tsv" --out "runs/$RUN_ID" --tag "var_${STYLE}" --latency-n 50
done
bash slurm/next_stage.sh
