#!/bin/bash
#SBATCH --job-name=xettat-guard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
# CPU-only: build the English-word guard from the 20k English list and the full Azerbaijani training mix.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python scripts/build_english_guard.py --en data/english/google-20k.txt --corpus data/mix/train.txt \
  --out src/xettat/english_guard.txt --copy-to web/shared/english-guard.txt extension/shared/english-guard.txt
