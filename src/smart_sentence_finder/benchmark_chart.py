from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Sequence


DISPLAY_NAME_MAP = {
    "Snowflake/snowflake-arctic-embed-l-v2.0": "Snowflake Arctic-Embed-L v2",
    "thenlper/gte-large": "GTE Large",
    "Qwen/Qwen3-Embedding-4B": "Qwen3 Embedding 4B",
    "google/embeddinggemma-300m": "EmbeddingGemma 300M",
    "NovaSearch/stella_en_1.5B_v5": "Stella EN 1.5B v5",
}


@dataclass(frozen=True)
class BenchmarkChartPaths:
    """Filesystem paths for generated benchmark chart assets."""

    svg: Path
    png: Path | None


def pretty_model_name(model_name: str) -> str:
    """Return a presentation-friendly label for a model."""

    return DISPLAY_NAME_MAP.get(model_name, model_name.split("/")[-1])


def format_params(param_count: int) -> str:
    """Format a parameter count for display."""

    if param_count >= 1_000_000_000:
        return f"{param_count / 1_000_000_000:.1f}B params"
    return f"{param_count / 1_000_000:.1f}M params"


def nice_axis_max(max_value: float) -> float:
    """Round the axis maximum to a presentation-friendly value."""

    step = 0.05
    return max(step, math.ceil(max_value / step) * step)


def _default_subtitle(rows: Sequence[dict[str, Any]]) -> str:
    sentence_count = int(rows[0]["n_sentences"])
    return f"{sentence_count} sentences benchmarked on the current corpus | higher is better"


def build_benchmark_svg(
    rows: Sequence[dict[str, Any]],
    title: str = "Model Silhouette Score Comparison",
    subtitle: str | None = None,
) -> str:
    """Render a silhouette-score comparison chart as SVG."""

    if not rows:
        raise ValueError("No benchmark rows were provided")

    sorted_rows = sorted((dict(row) for row in rows), key=lambda row: float(row["silhouette"]), reverse=True)
    chart_subtitle = subtitle or _default_subtitle(sorted_rows)

    max_label_chars = max(len(pretty_model_name(str(row["model_name"]))) for row in sorted_rows)
    outer_pad = 48
    label_area_width = max(360, min(560, max_label_chars * 18))
    value_column_width = 130
    chart_width = 1120
    width = outer_pad * 2 + label_area_width + chart_width + value_column_width + 80
    row_gap = 26
    bar_height = 84
    margin_left = outer_pad + label_area_width + 36
    margin_top = 240
    total_bars_height = len(sorted_rows) * bar_height + max(len(sorted_rows) - 1, 0) * row_gap
    chart_height = total_bars_height
    chart_bottom = margin_top + chart_height
    height = max(980, chart_bottom + 170)
    margin_right = width - (margin_left + chart_width)
    axis_max = nice_axis_max(max(float(row["silhouette"]) for row in sorted_rows))
    tick_step = 0.05
    best_model_name = str(sorted_rows[0]["model_name"])

    grid_lines: list[str] = []
    tick = 0.0
    while tick <= axis_max + 1e-9:
        x = margin_left + (tick / axis_max) * chart_width
        grid_lines.append(
            f'<line x1="{x:.1f}" y1="{margin_top - 8}" x2="{x:.1f}" y2="{chart_bottom + 16}" '
            'stroke="#d8d1c6" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{x:.1f}" y="{chart_bottom + 52}" text-anchor="middle" '
            'font-size="22" fill="#675f53">%.2f</text>' % tick
        )
        tick += tick_step

    bars: list[str] = []
    label_x = margin_left - 28
    score_x = margin_left + chart_width + 22
    for index, row in enumerate(sorted_rows):
        y = margin_top + index * (bar_height + row_gap)
        bar_width = (float(row["silhouette"]) / axis_max) * chart_width
        model_name = pretty_model_name(str(row["model_name"]))
        full_model_name = str(row["model_name"])
        fill = "#0f766e" if full_model_name == best_model_name else "#6b8aa6"

        bars.append(
            f'<text x="{label_x}" y="{y + 34}" text-anchor="end" font-size="28" '
            f'font-weight="700" fill="#0f172a">{escape(model_name)}</text>'
        )
        bars.append(
            f'<text x="{label_x}" y="{y + 66}" text-anchor="end" font-size="20" '
            'fill="#625a50">%s</text>' % escape(format_params(int(row["param_count"])))
        )
        bars.append(
            f'<rect x="{margin_left}" y="{y}" width="{chart_width:.1f}" height="{bar_height}" '
            'rx="20" fill="#ece6dc" />'
        )
        bars.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" '
            f'rx="20" fill="{fill}" />'
        )
        bars.append(
            f'<text x="{score_x}" y="{y + 53}" font-size="28" '
            'font-weight="700" fill="#111827">%.3f</text>' % float(row["silhouette"])
        )

        if full_model_name == best_model_name:
            badge_x = min(margin_left + bar_width - 104, margin_left + chart_width - 120)
            bars.append(
                f'<rect x="{badge_x:.1f}" y="{y + 24}" width="94" height="36" rx="18" fill="#dff4ed" />'
            )
            bars.append(
                f'<text x="{badge_x + 47:.1f}" y="{y + 48}" text-anchor="middle" '
                'font-size="18" font-weight="700" fill="#0f766e">best</text>'
            )

    footer_y = height - 84
    note = (
        "Silhouette score is an unsupervised clustering proxy. "
        "Higher is better for this benchmark, not necessarily for retrieval."
    )

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{escape(title)}</title>
  <desc id="desc">{escape(chart_subtitle)}</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f7f3eb" />
      <stop offset="100%" stop-color="#efe8dc" />
    </linearGradient>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#bg)" />
  <rect x="48" y="42" width="{width - 96}" height="{height - 84}" rx="36" fill="#fbfaf7" stroke="#d9d1c3" stroke-width="2" />
  <text x="86" y="114" font-size="56" font-weight="700" fill="#111827">{escape(title)}</text>
  <text x="86" y="162" font-size="24" fill="#5b554c">{escape(chart_subtitle)}</text>
  {''.join(grid_lines)}
  {''.join(bars)}
  <text x="86" y="{footer_y}" font-size="20" fill="#625a50">{escape(note)}</text>
</svg>
'''


def _try_png_conversion(svg_path: Path, png_path: Path) -> Path | None:
    """Best-effort conversion from SVG to PNG using common command-line tools."""

    converters: list[list[str]] = []

    if shutil.which("rsvg-convert"):
        converters.append(["rsvg-convert", "-o", str(png_path), str(svg_path)])
    if shutil.which("magick"):
        converters.append(["magick", str(svg_path), str(png_path)])
    if shutil.which("convert"):
        converters.append(["convert", str(svg_path), str(png_path)])
    if shutil.which("inkscape"):
        converters.append(
            ["inkscape", str(svg_path), "--export-type=png", f"--export-filename={png_path}"]
        )
    if shutil.which("ffmpeg"):
        converters.append(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(svg_path), "-frames:v", "1", str(png_path)]
        )

    for command in converters:
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception:
            if png_path.exists():
                png_path.unlink()
            continue
        if png_path.exists():
            return png_path

    return None


def save_benchmark_chart_assets(
    rows: Sequence[dict[str, Any]],
    *,
    svg_path: Path,
    title: str = "Model Silhouette Score Comparison",
    subtitle: str | None = None,
    png_path: Path | None = None,
) -> BenchmarkChartPaths:
    """Write chart assets to disk and optionally create a PNG copy."""

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg = build_benchmark_svg(rows, title=title, subtitle=subtitle)
    svg_path.write_text(svg, encoding="utf-8")

    generated_png: Path | None = None
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        generated_png = _try_png_conversion(svg_path, png_path)

    return BenchmarkChartPaths(svg=svg_path, png=generated_png)
