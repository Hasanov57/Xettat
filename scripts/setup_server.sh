#!/bin/bash
# One-time setup on the AI Academy GPU server (run in the JupyterLab terminal, inside the repo).
set -euo pipefail
cd "$(dirname "$0")/.."
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
mkdir -p logs data/processed
echo "Setup done. Next: sbatch slurm/test_gpu.sbatch"
