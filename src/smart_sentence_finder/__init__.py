"""Lightweight package init to avoid heavy imports at import time.

Modules should be imported explicitly, e.g.:
    from smart_sentence_finder.embedding import load_embedder
    from smart_sentence_finder.text import segment_text

This keeps optional deps (like pysbd) from being required unless the
corresponding module is actually used.
"""

__all__: list[str] = []
