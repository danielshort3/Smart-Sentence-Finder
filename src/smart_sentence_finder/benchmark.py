from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from .embedding import load_embedder


@dataclass
class BenchmarkResult:
    model_name: str
    best_k: int
    silhouette: float
    n_sentences: int
    dim: int
    param_count: int
    silhouette_per_million_params: float


def embed_sentences(
    model_name: str,
    sentences: List[str],
    batch_size: int = 16,
    max_length: int = 512,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    with load_embedder(model_name, device) as embedder:
        try:
            param_count = sum(p.numel() for p in embedder.model.parameters())
        except Exception:
            param_count = 0
        vecs = embedder.encode(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            normalize=True,
            show_progress=True,
            progress_desc=f"Embed {model_name}",
        )
        return vecs.to(torch.float32).cpu(), int(param_count)


def compute_best_silhouette(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    metric: str = "cosine",
    random_state: int = 42,
) -> Tuple[int, float]:
    best_k = k_min
    best_score = -1.0
    for k in tqdm(
        range(k_min, k_max + 1),
        total=(k_max - k_min + 1),
        desc="Silhouette k",
        leave=False,
        dynamic_ncols=True,
    ):
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
        score = silhouette_score(X, labels, metric=metric)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, float(best_score)


def benchmark_models(
    models: List[str],
    sentences: List[str],
    *,
    max_sentences: int = 1000,
    batch_size: int = 16,
    max_length: int = 512,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> List[BenchmarkResult]:
    if max_sentences and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    results: List[BenchmarkResult] = []
    for model_name in tqdm(models, desc="Models", leave=False, dynamic_ncols=True):
        vecs, param_count = embed_sentences(
            model_name,
            sentences,
            batch_size=batch_size,
            max_length=max_length,
        )
        # Ensure float32 before numpy conversion (avoid bfloat16 issues)
        X = vecs.to(torch.float32).detach().cpu().numpy()
        # Free tensor memory ASAP
        del vecs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        best_k, sil = compute_best_silhouette(X, k_min=k_min, k_max=k_max, random_state=random_state)
        results.append(
            BenchmarkResult(
                model_name=model_name,
                best_k=best_k,
                silhouette=sil,
                n_sentences=len(sentences),
                dim=X.shape[1],
                param_count=int(param_count),
                silhouette_per_million_params=float(sil / max(param_count / 1_000_000.0, 1e-9)),
            )
        )
    # Sort by silhouette descending
    results.sort(key=lambda r: r.silhouette, reverse=True)
    return results
