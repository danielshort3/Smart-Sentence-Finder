from __future__ import annotations

import gc
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


class Embedder:
    def __init__(self, model, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def __enter__(self) -> "Embedder":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @torch.inference_mode()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> torch.Tensor:
        # Sentence-Transformers model path
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            if isinstance(self.model, SentenceTransformer):
                vecs = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=normalize,
                )
                return vecs.to(self.device, dtype=torch.float32)
        except Exception:
            pass

        # HF Transformers path
        outputs: List[torch.Tensor] = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=(len(texts) + batch_size - 1) // batch_size,
                desc=progress_desc or "Embedding",
            )
        for i in iterator:
            batch = texts[i : i + batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}

            out = self.model(**toks)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                last_hidden = out.last_hidden_state
                mask = toks.get("attention_mask", torch.ones_like(last_hidden[:, :, 0]))
                mask = mask.unsqueeze(-1).type_as(last_hidden)
                summed = (last_hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                emb = summed / counts

            if normalize:
                emb = F.normalize(emb, p=2, dim=1)

            outputs.append(emb.to(torch.float32))

        return torch.cat(outputs, dim=0)

    def close(self):
        try:
            # Move off GPU and delete references
            if hasattr(self.model, "to"):
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            gc.collect()


def _get_hf_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.getenv(name)
        if val:
            return val
    return None


def _st_embedder(model_name: str, device: torch.device) -> Embedder | None:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    try:
        token = _get_hf_token()
        try:
            model = SentenceTransformer(model_name, token=token)
        except TypeError:
            # Older versions
            model = SentenceTransformer(model_name, use_auth_token=token)
        model.to(device)
    except Exception:
        return None

    return Embedder(model=model, tokenizer=None, device=device)


def _hf_embedder(model_name: str, device: torch.device) -> Embedder:
    from transformers import AutoModel, AutoTokenizer

    token = _get_hf_token()
    # New-style token arg first, then fallback to deprecated use_auth_token
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(model_name, token=token, trust_remote_code=True)
    except TypeError:
        model = AutoModel.from_pretrained(model_name, use_auth_token=token, trust_remote_code=True)
    model.to(device)
    model.eval()

    return Embedder(model=model, tokenizer=tokenizer, device=device)


def load_embedder(model_name: str, device: Optional[torch.device] = None) -> Embedder:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st = _st_embedder(model_name, device)
    if st is not None:
        return st

    return _hf_embedder(model_name, device)
