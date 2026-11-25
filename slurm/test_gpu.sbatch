#!/bin/bash
#SBATCH --job-name=xettat-test
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
# Debug slice: GPU check + full unit test suite (incl. torch tests that skip on laptops).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q -rs
