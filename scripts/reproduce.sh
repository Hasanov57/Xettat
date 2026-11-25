#!/bin/bash
# One command that reproduces the headline numbers from a clean checkout + saved checkpoints.
#   bash scripts/reproduce.sh            # uses checkpoints listed in runs/HEADLINE
# Checkpoints live in $XETTAT_CKPT_ROOT (default /data/teams/team14/checkpoints).
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT_ROOT=${XETTAT_CKPT_ROOT:-/data/teams/team14/checkpoints}
if [ ! -s data/processed/test.txt ]; then
  python -m xettat.data.prepare --source wiki --out data/processed --seed 42
fi
python -m xettat.run_baselines --data data/processed --split test --out runs/baselines
if [ -f runs/HEADLINE ]; then
  while read -r RUN_ID; do
    [ -z "$RUN_ID" ] && continue
    python -m xettat.evaluate --ckpt "$CKPT_ROOT/$RUN_ID/best.pt" --data data/processed --split test --out "runs/$RUN_ID"
  done < runs/HEADLINE
fi
python -m xettat.report_table --runs runs --split test --out report/tables/results.md
