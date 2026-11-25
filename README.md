# Xəttat

**Context-aware Azerbaijani diacritic restoration** — turns text typed on an English
keyboard (`uzumun suyu uzume sicradi`) into correct Azerbaijani (`üzümün suyu üzümə sıçradı`).

DLE-AI-202 Deep Learning Final Project · Cohort II 2026 · Track 3 (Industry Product) · Team 14

Product: Chrome extension (main), website (info + demo), Docker API.
Design: [`docs/superpowers/specs/2026-11-25-xettat-design.md`](docs/superpowers/specs/2026-11-25-xettat-design.md)

## Setup

```bash
git clone https://github.com/Hasanov57/Xettat.git && cd Xettat
bash scripts/setup_server.sh        # torch (cu128) + requirements + editable install
pytest -q                           # unit tests
```

## Data

Training pairs are generated automatically: correct Azerbaijani sentences
(Azerbaijani Wikipedia, `wikimedia/wikipedia` `20231101.az`, CC BY-SA 4.0) are stripped
of diacritics. Split is done at article level and frozen before experiments.

```bash
python -m xettat.data.prepare --source wiki --out data/processed --seed 42
```

## Train

```bash
sbatch slurm/train.sbatch configs/bilstm.yaml bilstm_v1
```

## Reproduce headline results

```bash
bash scripts/reproduce.sh
```

Server step-by-step guide (Azerbaijani): [`docs/SERVER_GUIDE_AZ.md`](docs/SERVER_GUIDE_AZ.md)
