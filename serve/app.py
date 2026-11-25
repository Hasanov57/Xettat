"""Xəttat HTTP API (Track 3 deployment).

    MODEL_FAST=checkpoints/bilstm_v1/best.pt MODEL_ACCURATE=checkpoints/byt5_v1/best.pt \
        uvicorn serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import threading
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from xettat.restorer import Restorer

MAX_CHARS = 20_000
PATHS = {"fast": os.environ.get("MODEL_FAST"), "accurate": os.environ.get("MODEL_ACCURATE")}

app = FastAPI(title="Xəttat API", version="0.1.0",
              description="Azerbaijani diacritic restoration: seher → səhər")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

_models: dict[str, Restorer] = {}
_lock = threading.Lock()


def get_model(name: str) -> tuple[str, Restorer]:
    if not PATHS.get(name):
        name = "fast" if PATHS.get("fast") else "accurate"
    if not PATHS.get(name):
        raise HTTPException(503, "No model configured: set MODEL_FAST and/or MODEL_ACCURATE")
    with _lock:
        if name not in _models:
            _models[name] = Restorer.from_checkpoint(PATHS[name], device=os.environ.get("DEVICE", "cpu"))
    return name, _models[name]


class RestoreIn(BaseModel):
    text: str
    model: Literal["fast", "accurate"] = "fast"


class RestoreOut(BaseModel):
    text: str
    changed: int
    ms: float
    model: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models": {k: bool(v) for k, v in PATHS.items()}, "loaded": sorted(_models)}


@app.post("/restore", response_model=RestoreOut)
def restore(body: RestoreIn) -> RestoreOut:
    if len(body.text) > MAX_CHARS:
        raise HTTPException(413, f"Text too long: max {MAX_CHARS} characters")
    name, model = get_model(body.model)
    t0 = time.perf_counter()
    out = model.restore(body.text)
    ms = 1000 * (time.perf_counter() - t0)
    return RestoreOut(text=out, changed=sum(a != b for a, b in zip(body.text, out)), ms=round(ms, 2), model=name)
