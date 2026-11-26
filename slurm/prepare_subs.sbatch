#!/bin/bash
#SBATCH --job-name=xettat-subs
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# CPU-only. Adds OpenSubtitles (dialogue) and rebuilds data/mix = Wikipedia + HPLT + subtitles.
# Run it BEFORE bilstm_mix starts training. Does not advance the pipeline queue.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python -m xettat.data.prepare --source opensubs --out data/subs --seed 42 --doc-min-az-ratio 0.08 --val-frac 0.02 --test-frac 0.02
python -m xettat.data.mix --sources data/processed data/hplt data/subs --out data/mix --seed 42
python -m xettat.run_baselines --data data/subs --split val --out runs/baselines_subs
