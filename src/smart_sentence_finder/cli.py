import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List

from .text import segment_text, clean_sentences
from .search import process
from .benchmark import benchmark_models


DEFAULT_MODELS: List[str] = [
    "Qwen/Qwen3-Embedding-4B",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "NovaSearch/stella_en_1.5B_v5",
    "thenlper/gte-large",
    "google/embeddinggemma-300m",
]


def normalize_text(s: str) -> str:
    return s.replace("\n", " ").replace("-", " ").replace("_", " ")


def add_common_text_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--file", required=True, help="Path to the input text file")
    p.add_argument("--chars-per-chunk", type=int, default=10_000, help="Character budget per chunk for segmentation")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Sentence Finder")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # rank subcommand
    rank = sub.add_parser("rank", help="Rank sentences against a query using cosine similarity")
    add_common_text_args(rank)
    rank.add_argument("--query", required=True, help="Query to search for relevant sentences")
    rank.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Embedding models to use")
    rank.add_argument("--top", type=int, default=5, help="Number of top sentences to show")
    rank.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    rank.add_argument("--max-length", type=int, default=512, help="Tokenizer max length")
    rank.add_argument("--output-dir", default="output", help="Directory to write results (created if missing)")

    # benchmark subcommand
    bench = sub.add_parser("benchmark", help="Benchmark models by silhouette score on sentence embeddings")
    add_common_text_args(bench)
    bench.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Embedding models to benchmark")
    bench.add_argument("--max-sentences", type=int, default=1000, help="Limit sentences for benchmarking")
    bench.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    bench.add_argument("--max-length", type=int, default=512, help="Tokenizer max length")
    bench.add_argument("--k-min", type=int, default=2, help="Minimum clusters for silhouette search")
    bench.add_argument("--k-max", type=int, default=10, help="Maximum clusters for silhouette search")
    bench.add_argument("--output-dir", default="output", help="Directory to write results (created if missing)")

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")

    text = normalize_text(path.read_text(encoding="utf-8"))
    sentences = clean_sentences(segment_text(text, chars_per_chunk=args.chars_per_chunk))

    # Decide command mode
    cmd = args.cmd
    if cmd is None:
        print("Please specify a subcommand: rank or benchmark")
        return 2

    models = args.models if args.models else DEFAULT_MODELS

    if cmd == "rank":
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(args.file).stem
        out_txt = out_dir / f"rank_{base}_{ts}.txt"

        with out_txt.open("w", encoding="utf-8") as f:
            f.write(f"Query: {args.query}\n\n")
            for model_name in models:
                res = process(
                    model_name,
                    args.query,
                    sentences,
                    n=args.top or 5,
                    batch_size=args.batch_size or 32,
                    max_length=args.max_length or 512,
                )
                f.write(f"Model: {model_name}\n")
                for i, (sent, score) in enumerate(res["top"], start=1):
                    f.write(f"{i}. {float(score):.4f} | {sent}\n")
                f.write("\n")
        print(f"Saved ranking results to {out_txt}")
        return 0

    if cmd == "benchmark":
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(args.file).stem
        results = benchmark_models(
            models,
            [s for s in sentences if len(s.split()) > 5],
            max_sentences=args.max_sentences,
            batch_size=args.batch_size,
            max_length=args.max_length,
            k_min=args.k_min,
            k_max=args.k_max,
        )
        print("Model Silhouette Scores (higher is better):")
        for r in results:
            print(
                f"- {r.model_name}: silhouette={r.silhouette:.4f} | per_param={r.silhouette/ max(r.param_count,1):.6e} | per_Mparam={r.silhouette_per_million_params:.6e} "
                f"(best_k={r.best_k}, n={r.n_sentences}, dim={r.dim}, params={r.param_count/1_000_000:.2f}M)"
            )
        # Save CSV
        out_csv = out_dir / f"benchmark_{base}_{ts}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["model_name", "silhouette", "silhouette_per_million_params", "best_k", "n_sentences", "dim", "param_count"])
            for r in results:
                writer.writerow([r.model_name, f"{r.silhouette:.6f}", f"{r.silhouette_per_million_params:.6e}", r.best_k, r.n_sentences, r.dim, r.param_count])
        # Save JSON
        out_json = out_dir / f"benchmark_{base}_{ts}.json"
        with out_json.open("w", encoding="utf-8") as jf:
            json.dump(
                [
                    {
                        "model_name": r.model_name,
                        "silhouette": r.silhouette,
                        "silhouette_per_million_params": r.silhouette_per_million_params,
                        "best_k": r.best_k,
                        "n_sentences": r.n_sentences,
                        "dim": r.dim,
                        "param_count": r.param_count,
                    }
                    for r in results
                ],
                jf,
                indent=2,
            )
        print(f"Saved benchmark results to {out_csv} and {out_json}")
        return 0

    raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
