#!/bin/bash
# Pops the next command from .pipeline_queue and submits it. Called at the end of each successful stage.
cd "$(dirname "$0")/.."
Q=.pipeline_queue
[ -s "$Q" ] || { echo "[pipeline] queue empty - all stages done"; exit 0; }
NEXT=$(head -n 1 "$Q")
tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
echo "[pipeline] submitting: sbatch $NEXT"
# shellcheck disable=SC2086
sbatch $NEXT
