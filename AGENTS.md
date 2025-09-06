# Repository Guidelines

## Project Structure & Module Organization
- src/smart_sentence_finder: core package
  - cli.py: CLI (rank, benchmark)
  - text.py: chunking, sentence segmentation, cleaning
  - search.py: query–sentence ranking via cosine similarity
  - embedding.py: generic HF/ST embedder loader
  - benchmark.py: KMeans + silhouette scoring
- data: sample texts (e.g., alice_in_wonderland.txt)
- Dockerfile, requirements.txt, pyproject.toml, README.md, AGENTS.md

## Build, Test, and Development Commands
- Create env and install (dev):
  - python -m venv .venv && source .venv/bin/activate
  - pip install -e .[dev]
- Run locally:
  - python -m smart_sentence_finder.cli rank --file data/alice_in_wonderland.txt --query "She wonders about things." --top 5
  - python -m smart_sentence_finder.cli benchmark --file data/alice_in_wonderland.txt --max-sentences 800
- Docker (GPU):
  - docker build -t smart-sentence-finder .
  - docker run --rm -it --gpus all -v "$PWD/data:/app/data" -v "$PWD/models:/models" smart-sentence-finder benchmark --file /app/data/alice_in_wonderland.txt

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indents, type hints required for new code.
- Names: modules/functions/vars use snake_case; Classes use PascalCase; constants UPPER_SNAKE_CASE.
- Use docstrings for public functions; keep functions short and cohesive.
- Lint/format: ruff (run: ruff check .; ruff format .).

## Testing Guidelines
- Framework: pytest. Place tests under tests/ named test_*.py.
- Write unit tests for text segmentation, embedding, and ranking; prefer small fixtures.
- Run tests: pytest -q

## Commit & Pull Request Guidelines
- Commits: imperative present (e.g., "Add benchmark CLI"); group related changes.
- Prefer Conventional Commits (feat, fix, docs, refactor) where practical.
- PRs: include a clear description, linked issues, steps to reproduce/validate, and any screenshots or logs.
- Keep PRs focused; include docs/README updates for user-facing changes.

## Security & Configuration Tips
- Docker caches models under /models; persist by mounting -v "$PWD/models:/models".
- Do not hardcode API keys/tokens. Use environment variables and avoid committing secrets.
