import argparse
import json
from pathlib import Path

import numpy as np

from smart_sentence_finder.text import segment_text, clean_sentences
from smart_sentence_finder.embedding import load_embedder


def main() -> int:
    p = argparse.ArgumentParser(description="Precompute Alice in Wonderland sentence embeddings for a model")
    p.add_argument("--file", default="data/alice_in_wonderland.txt", help="Path to Alice text file")
    p.add_argument("--model", default="Snowflake/snowflake-arctic-embed-l-v2.0", help="Embedding model name")
    p.add_argument("--output-dir", default="artifacts/snowflake_arctic_v2", help="Output directory for artifacts")
    p.add_argument("--chars-per-chunk", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    args = p.parse_args()

    text_path = Path(args.file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = text_path.read_text(encoding="utf-8").replace("\n", " ").replace("-", " ").replace("_", " ")
    sentences = clean_sentences(segment_text(text, chars_per_chunk=args.chars_per_chunk))
    sentences = [s for s in sentences if len(s.split()) > 5]

    with load_embedder(args.model) as embedder:
        vecs = embedder.encode(sentences, batch_size=args.batch_size, max_length=args.max_length, normalize=True, show_progress=True, progress_desc=f"Embed {args.model}")

    # Save
    (out_dir / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "embeddings.npy", vecs.cpu().numpy())
    meta = {
        "model_name": args.model,
        "n_sentences": len(sentences),
        "dim": int(vecs.shape[1]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    print(f"Saved {len(sentences)} sentences and embeddings to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
