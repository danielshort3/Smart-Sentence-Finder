from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smart_sentence_finder.benchmark_chart import save_benchmark_chart_assets


def load_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    """Load benchmark rows from a JSON or CSV export."""

    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list of benchmark rows in {path}")
        return rows

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = list(reader)
        numeric_fields = {
            "silhouette": float,
            "silhouette_per_million_params": float,
            "best_k": int,
            "n_sentences": int,
            "dim": int,
            "param_count": int,
        }
        parsed_rows: list[dict[str, Any]] = []
        for row in rows:
            parsed = dict(row)
            for key, caster in numeric_fields.items():
                parsed[key] = caster(parsed[key])
            parsed_rows.append(parsed)
        return parsed_rows

    raise ValueError(f"Unsupported benchmark format: {path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a benchmark silhouette comparison chart as SVG")
    parser.add_argument("benchmark_file", type=Path, help="Path to benchmark JSON or CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path. Defaults to the benchmark file stem with _silhouette_scores.svg",
    )
    parser.add_argument(
        "--title",
        default="Model Silhouette Score Comparison",
        help="Chart title",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_benchmark_rows(args.benchmark_file)
    output_path = args.output or args.benchmark_file.with_name(
        f"{args.benchmark_file.stem}_silhouette_scores.svg"
    )

    chart_paths = save_benchmark_chart_assets(
        rows,
        svg_path=output_path,
        title=args.title,
    )
    print(chart_paths.svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
