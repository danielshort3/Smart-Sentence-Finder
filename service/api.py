import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import os
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from smart_sentence_finder.embedding import load_embedder


MODEL_NAME = os.getenv("MODEL_NAME", "Snowflake/snowflake-arctic-embed-l-v2.0")
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts/snowflake_arctic_v2"))
SENT_PATH = Path(os.getenv("SENT_PATH", ARTIFACT_DIR / "sentences.json"))
EMB_PATH = Path(os.getenv("EMB_PATH", ARTIFACT_DIR / "embeddings.npy"))
META_PATH = Path(os.getenv("META_PATH", ARTIFACT_DIR / "meta.json"))

app = FastAPI(title="Smart Sentence Finder API")


class RankRequest(BaseModel):
    query: str
    top: int = 5


class _State:
    sentences: List[str] | None = None
    embeddings: torch.Tensor | None = None
    embedder = None
    meta: dict | None = None


state = _State()


def _host_from_header(value: str | None) -> str | None:
    if not value:
        return None
    try:
        url = urlparse(value)
        return url.netloc or url.path  # handle bare hosts
    except Exception:
        return None


@app.middleware("http")
async def origin_guard(request: Request, call_next):
    # Allow health unconditionally
    if request.url.path == "/health":
        return await call_next(request)

    # OPTIONS passthrough (preflight)
    if request.method.upper() == "OPTIONS":
        return await call_next(request)

    allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    # If no allowlist configured, do not block (backward compatible)
    if not allowed:
        return await call_next(request)

    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    host = request.headers.get("host")
    xf_host = request.headers.get("x-forwarded-host")
    origin_host = _host_from_header(origin)
    referer_host = _host_from_header(referer)
    req_host = xf_host or host

    allowed_hosts = set(_host_from_header(a) or a for a in allowed)

    # Debug bypass via header token
    dbg_token = os.getenv("DEBUG_TOKEN")
    if dbg_token and request.headers.get("x-debug-token") == dbg_token:
        return await call_next(request)

    # Permit if any known host matches allowlist
    candidates = {h for h in (origin_host, referer_host, req_host) if h}
    if allowed_hosts.intersection(candidates):
        return await call_next(request)

    return JSONResponse({"error": "forbidden", "detail": "origin not allowed"}, status_code=403)


@app.on_event("startup")
def _startup() -> None:
    # Load artifacts
    import json

    if not SENT_PATH.exists() or not EMB_PATH.exists():
        raise RuntimeError(f"Artifacts not found. Expected {SENT_PATH} and {EMB_PATH}")

    state.sentences = json.loads(Path(SENT_PATH).read_text(encoding="utf-8"))
    arr = np.load(EMB_PATH)
    state.embeddings = torch.from_numpy(arr).to(torch.float32)
    # Load meta if present
    if META_PATH.exists():
        try:
            state.meta = json.loads(META_PATH.read_text())
        except Exception:
            state.meta = None
    else:
        state.meta = None

    # Load embedder (CPU for Lambda; GPU if available elsewhere)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state.embedder = load_embedder(MODEL_NAME, device)

    # Validate that artifacts match the configured model and shapes
    meta = state.meta or {}
    dim = int(state.embeddings.shape[1])
    if not meta:
        raise RuntimeError(
            "Artifact metadata missing (meta.json). Rebuild artifacts with the same MODEL_NAME used at runtime."
        )
    if meta.get("model_name") != MODEL_NAME:
        raise RuntimeError(
            f"Artifact model mismatch. meta.model_name={meta.get('model_name')} != runtime MODEL_NAME={MODEL_NAME}"
        )
    if int(meta.get("dim", dim)) != dim:
        raise RuntimeError(
            f"Artifact embedding dimension mismatch. meta.dim={meta.get('dim')} vs file dim={dim}"
        )
    if int(meta.get("n_sentences", len(state.sentences))) != len(state.sentences):
        raise RuntimeError(
            f"Artifact sentence count mismatch. meta.n_sentences={meta.get('n_sentences')} vs sentences={len(state.sentences)}"
        )


@app.on_event("shutdown")
def _shutdown() -> None:
    try:
        if state.embedder is not None:
            state.embedder.close()
    finally:
        state.embedder = None


@app.get("/health")
def health() -> dict:
    dim = int(state.embeddings.shape[1]) if state.embeddings is not None else None
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "sentences": len(state.sentences or []),
        "dim": dim,
        "artifact_meta": state.meta,
    }


@app.post("/rank")
def rank(req: RankRequest) -> dict:
    if not req.query:
        return {"error": "query is required"}

    assert state.embeddings is not None and state.sentences is not None and state.embedder is not None
    # Encode query
    q = state.embedder.encode([req.query], batch_size=1, max_length=512, normalize=True)[0]
    # Cosine with precomputed sentence embeddings
    scores = F.cosine_similarity(state.embeddings, q.unsqueeze(0), dim=1).to(torch.float32).cpu().numpy()
    # Top-k
    k = max(1, min(req.top, len(state.sentences)))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    results = [
        {"rank": i + 1, "score": float(scores[idx]), "sentence": state.sentences[idx]}
        for i, idx in enumerate(top_idx)
    ]
    return {"model": MODEL_NAME, "query": req.query, "top": results}
