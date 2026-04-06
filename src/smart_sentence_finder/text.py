import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List

import pysbd
from tqdm import tqdm


def split_by_character_count(text: str, chars_per_chunk: int) -> List[str]:
    """Split text into chunks close to the given size, breaking at periods.

    Falls back to extending until the next period if none is found in the window.
    """
    total_chars = len(text)
    chunks: List[str] = []
    start = 0

    while start < total_chars:
        end = min(start + chars_per_chunk, total_chars)

        if end < total_chars:
            while end > start and text[end - 1] != ".":
                end -= 1

            if end == start:
                while end < total_chars and text[end - 1] != ".":
                    end += 1

        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end

    return chunks


def process_chunk(chunk: str) -> List[str]:
    seg = pysbd.Segmenter(language="en", clean=False)
    return seg.segment(chunk)


def segment_text(text: str, chars_per_chunk: int = 10_000) -> List[str]:
    """Segment a long string into sentences using chunking and threading when helpful."""
    chunks = split_by_character_count(text, chars_per_chunk)

    if len(text) > chars_per_chunk * 2:
        num_threads = max(1, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            sentences = [
                s
                for f in tqdm(as_completed(futures), total=len(futures), leave=False, dynamic_ncols=True)
                for s in f.result()
            ]
    else:
        sentences = process_chunk(text)

    return sentences


def clean_sentences(sentences: Iterable[str]) -> List[str]:
    cleaned_sentences: List[str] = []
    modification_count = 0

    for sentence in tqdm(sentences, leave=False, dynamic_ncols=True):
        trimmed_sentence = sentence.strip()
        cleaned_sentence = re.sub(r"\s+", " ", trimmed_sentence)

        if sentence != cleaned_sentence:
            modification_count += 1

        cleaned_sentences.append(cleaned_sentence)

    print(f"{modification_count} sentences cleaned.")
    return cleaned_sentences

