#!/bin/bash
#SBATCH --job-name=xettat-data2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# CPU-only. Adds conversational/web Azerbaijani (HPLT web text + OpenSubtitles dialogue) and builds data/mix.
# Can run while a GPU training job is running. Does NOT advance the pipeline queue.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
python -m xettat.data.prepare --source hplt:3 --out data/hplt --seed 42 --doc-min-az-ratio 0.08 --max-train 3000000 \
  || echo "!! HPLT failed"
python -m xettat.data.prepare --source opensubs --out data/subs --seed 42 --doc-min-az-ratio 0.08 --val-frac 0.02 --test-frac 0.02 \
  || echo "!! OpenSubtitles failed (continuing without it)"
python -m xettat.data.mix --sources data/processed data/hplt data/subs --out data/mix --seed 42
for d in data/hplt data/subs; do [ -f $d/stats.json ] && python -m xettat.run_baselines --data $d --split val --out runs/baselines_$(basename $d); done
