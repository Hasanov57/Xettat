"""Resumable training for char-level taggers.

    python -m xettat.train --config configs/bilstm.yaml --run-id bilstm_v1 [--max-steps N] [--time-budget-min M]

Exit code 99 = stopped early by signal/time budget (resubmit to continue); 0 = finished.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from xettat.dataset import SentenceDataset, collate, worker_init
from xettat.infer import predict_texts
from xettat.metrics import score
from xettat.models import build_model
from xettat.text import IGNORE, strip
from xettat.utils import RunLogger, set_seed
from xettat.vocab import CharVocab

RESUBMIT = 99
_STOP = {"flag": False}


def _on_signal(signum, frame):  # noqa: ARG001
    _STOP["flag"] = True


def lr_lambda(warmup: int, total: int, floor: float = 0.05):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


def evaluate_val(model, vocab, targets, device, max_len) -> dict:
    model.eval()
    inputs = [strip(t) for t in targets]
    preds = predict_texts(model, vocab, inputs, device, max_len)
    model.train()
    m = score(inputs, preds, targets)
    return {f"val_{k}": v for k, v in m.items() if isinstance(v, float)}


def save(path: Path, **state) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--time-budget-min", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    tc, dc = cfg["train"], cfg["data"]
    run_id = args.run_id or Path(args.config).stem
    ckpt_dir = Path(os.environ.get("XETTAT_CKPT_ROOT", cfg.get("ckpt_root", "checkpoints"))) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(cfg.get("runs_root", "runs")) / run_id
    logger = RunLogger(run_dir)
    if not (run_dir / "config.yaml").exists():
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = CharVocab()
    max_len = dc["max_len"]
    model = build_model(cfg["model"], vocab.size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.01))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc.get("warmup_steps", 0), tc["max_steps"]))

    step, best, bad_evals, done = 0, -1.0, 0, False
    last = ckpt_dir / "last.pt"
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        step, best, bad_evals, done = st["step"], st["best"], st["bad_evals"], st.get("done", False)
        random.setstate(st["py_rng"])
        np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"])
        logger.log({"event": "resume", "from_step": step})
        print(f"resumed from step {step}")
    if done == "early_stop" or step >= tc["max_steps"]:
        print("run already finished")
        return 0

    n_params = sum(p.numel() for p in model.parameters())
    logger.log({"event": "start", "params": n_params, "device": str(device)})
    data_dir = Path(dc["dir"])
    train_ds = SentenceDataset(data_dir / "train.txt", vocab, max_len, augment=dc.get("augment", True),
                               seed=cfg.get("seed", 42) + step, limit=dc.get("train_limit"))
    val_targets = [l for l in (data_dir / "val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_targets = val_targets[: dc.get("val_subset", 5000)]
    nw = tc.get("num_workers", 4)
    loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=nw, collate_fn=collate,
                        worker_init_fn=worker_init if nw else None, drop_last=len(train_ds) > tc["batch_size"],
                        persistent_workers=nw > 0, pin_memory=device.type == "cuda")

    def state(**extra):
        return dict(model=model.state_dict(), optim=opt.state_dict(), sched=sched.state_dict(), step=step,
                    best=best, bad_evals=bad_evals, cfg=cfg, vocab=vocab.to_json(), py_rng=random.getstate(),
                    np_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), **extra)

    for s in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(s, _on_signal)
    _STOP["flag"] = False
    t0 = time.time()
    budget = args.time_budget_min * 60 if args.time_budget_min else None
    use_amp = device.type == "cuda"
    model.train()
    run_loss, run_n = 0.0, 0

    while True:
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device, non_blocking=True), mask.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, mask)
            sel = labels != IGNORE
            loss = F.binary_cross_entropy_with_logits(logits.float()[sel], labels[sel].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
            opt.step()
            sched.step()
            step += 1
            run_loss, run_n = run_loss + loss.item(), run_n + 1

            if step % tc.get("log_every", 100) == 0:
                rec = {"step": step, "loss": run_loss / run_n, "lr": sched.get_last_lr()[0]}
                if use_amp:
                    rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                logger.log(rec)
                run_loss, run_n = 0.0, 0

            finished = step >= tc["max_steps"]
            if step % tc["eval_every"] == 0 or finished:
                m = evaluate_val(model, vocab, val_targets, device, max_len)
                if m["val_word_acc"] > best:
                    best, bad_evals = m["val_word_acc"], 0
                    save(ckpt_dir / "best.pt", model=model.state_dict(), cfg=cfg, vocab=vocab.to_json(),
                         step=step, val=m)
                else:
                    bad_evals += 1
                logger.log({"step": step, **m, "best_val_word_acc": best})
                if bad_evals >= tc.get("patience", 10):
                    logger.log({"event": "early_stop", "step": step})
                    finished = "early_stop"

            if finished:
                save(last, **state(done=finished if isinstance(finished, str) else "max_steps"))
                logger.log({"event": "done", "step": step, "best_val_word_acc": best})
                return 0
            if step % tc["ckpt_every"] == 0:
                save(last, **state())
            if _STOP["flag"] or (budget and time.time() - t0 > budget):
                save(last, **state())
                logger.log({"event": "preempt", "step": step})
                return RESUBMIT


if __name__ == "__main__":
    sys.exit(main())
