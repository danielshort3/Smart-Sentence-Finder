from __future__ import annotations

import csv
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .benchmark import benchmark_models
from .cli import DEFAULT_MODELS, normalize_text
from .search import process
from .text import clean_sentences, segment_text


@dataclass
class PreparedText:
    """Container for the normalized text and derived sentence lists."""

    text: str
    sentences: list[str]
    rankable_sentences: list[str]


def _module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def get_flash_attn_version() -> str | None:
    """Return the installed FlashAttention version, if available."""

    try:
        import flash_attn

        return getattr(flash_attn, "__version__", "installed")
    except Exception:
        return None


def get_runtime_info(device: torch.device | None = None) -> dict[str, object]:
    """Collect notebook-friendly runtime details for display."""

    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(resolved_device),
        "gpu": gpu_name,
        "flash_attn": get_flash_attn_version() or "not installed",
        "ipywidgets_installed": _module_available("ipywidgets"),
        "hf_token_present": any(
            os.getenv(name) for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
        ),
        "embed_backend": os.getenv("SSF_EMBED_BACKEND", "(default)"),
        "models": list(DEFAULT_MODELS),
    }


def configure_notebook_progress() -> dict[str, Any]:
    """Configure cleaner notebook progress bars and suppress noisy library bars when possible."""

    if _module_available("ipywidgets"):
        from tqdm.notebook import tqdm as base_tqdm

        backend = "tqdm.notebook"
    else:
        from tqdm.auto import tqdm as base_tqdm

        backend = "tqdm.auto"

    def notebook_tqdm(*args, **kwargs):
        kwargs.setdefault("leave", False)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("smoothing", 0.1)
        return base_tqdm(*args, **kwargs)

    import smart_sentence_finder.benchmark as benchmark_mod
    import smart_sentence_finder.embedding as embedding_mod
    import smart_sentence_finder.search as search_mod
    import smart_sentence_finder.text as text_mod

    for module in (benchmark_mod, embedding_mod, search_mod, text_mod):
        module.tqdm = notebook_tqdm

    suppressed = {
        "huggingface_hub": False,
        "transformers": False,
    }
    try:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        suppressed["huggingface_hub"] = True
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        suppressed["transformers"] = True
    except Exception:
        pass

    return {
        "tqdm_backend": backend,
        "library_progress_suppressed": suppressed,
    }


def prepare_text_data(data_file: Path, chars_per_chunk: int = 10_000) -> PreparedText:
    """Load, normalize, segment, and clean a text file for ranking and benchmarking."""

    text = normalize_text(data_file.read_text(encoding="utf-8"))
    sentences = clean_sentences(segment_text(text, chars_per_chunk=chars_per_chunk))
    rankable_sentences = [sentence for sentence in sentences if len(sentence.split()) > 5]
    return PreparedText(text=text, sentences=sentences, rankable_sentences=rankable_sentences)


def rank_models_for_query(
    models: Sequence[str],
    query: str,
    sentences: Sequence[str],
    *,
    top_n: int = 5,
    batch_size: int = 32,
    max_length: int = 512,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Rank a query across multiple models and return notebook-friendly row payloads."""

    top1_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    rank_payload: list[dict[str, object]] = []

    for model_name in models:
        result = process(
            model_name,
            query,
            sentences,
            n=top_n,
            batch_size=batch_size,
            max_length=max_length,
            show_step_progress=False,
        )

        top_rows: list[dict[str, object]] = []
        for rank, (sentence, score) in enumerate(result["top"], start=1):
            row = {
                "model_name": model_name,
                "rank": rank,
                "score": float(score),
                "sentence": sentence,
            }
            rank_rows.append(row)
            top_rows.append(row)

        if top_rows:
            top1_rows.append(top_rows[0])

        rank_payload.append(
            {
                "model_name": model_name,
                "filtered": int(result["filtered"]),
                "dim": int(result["dim"]),
                "top": top_rows,
            }
        )

        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return top1_rows, rank_rows, rank_payload


def build_benchmark_rows(
    models: Sequence[str],
    sentences: Sequence[str],
    *,
    max_sentences: int = 1_000,
    batch_size: int = 16,
    max_length: int = 512,
    k_min: int = 2,
    k_max: int = 10,
) -> list[dict[str, object]]:
    """Run the repository benchmark and return plain rows for notebook display and export."""

    benchmark_results = benchmark_models(
        list(models),
        list(sentences),
        max_sentences=max_sentences,
        batch_size=batch_size,
        max_length=max_length,
        k_min=k_min,
        k_max=k_max,
    )

    return [
        {
            "model_name": result.model_name,
            "silhouette": result.silhouette,
            "silhouette_per_million_params": result.silhouette_per_million_params,
            "best_k": result.best_k,
            "n_sentences": result.n_sentences,
            "dim": result.dim,
            "param_count": result.param_count,
        }
        for result in benchmark_results
    ]


def save_rank_results(
    output_dir: Path,
    data_file: Path,
    run_ts: str,
    query: str,
    models: Sequence[str],
    rank_payload: Sequence[dict[str, object]],
    rank_rows: Sequence[dict[str, object]],
) -> tuple[Path, Path]:
    """Save ranking outputs in JSON and CSV formats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_json = output_dir / f"rank_{data_file.stem}_{run_ts}.json"
    rank_csv = output_dir / f"rank_{data_file.stem}_{run_ts}.csv"

    with rank_json.open("w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "query": query,
                "file": str(data_file),
                "models": list(models),
                "results": list(rank_payload),
            },
            file_obj,
            indent=2,
        )

    with rank_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["model_name", "rank", "score", "sentence"])
        writer.writeheader()
        writer.writerows(rank_rows)

    return rank_json, rank_csv


def save_benchmark_results(
    output_dir: Path,
    data_file: Path,
    run_ts: str,
    benchmark_rows: Sequence[dict[str, object]],
) -> tuple[Path, Path]:
    """Save benchmark outputs in JSON and CSV formats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_json = output_dir / f"benchmark_{data_file.stem}_{run_ts}.json"
    benchmark_csv = output_dir / f"benchmark_{data_file.stem}_{run_ts}.csv"

    with benchmark_json.open("w", encoding="utf-8") as file_obj:
        json.dump(list(benchmark_rows), file_obj, indent=2)

    with benchmark_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "model_name",
                "silhouette",
                "silhouette_per_million_params",
                "best_k",
                "n_sentences",
                "dim",
                "param_count",
            ],
        )
        writer.writeheader()
        writer.writerows(benchmark_rows)

    return benchmark_json, benchmark_csv
