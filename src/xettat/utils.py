from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class RunLogger:
    """Append-only JSONL metrics (+ TensorBoard scalars when available)."""

    def __init__(self, run_dir: str | Path) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "metrics.jsonl"
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(self.dir / "tb"))
        except Exception:
            pass

    def log(self, rec: dict) -> None:
        rec = {"time": time.time(), **rec}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self.tb is not None and "step" in rec:
            for k, v in rec.items():
                if isinstance(v, (int, float)) and k not in ("step", "time"):
                    self.tb.add_scalar(k, v, rec["step"])
            self.tb.flush()


def env_or(key: str, default: str) -> str:
    return os.environ.get(key, default)
