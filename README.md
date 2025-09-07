# Smart Sentence Finder

This repository contains a project aimed at finding the most relevant sentences to an input query using various pre-trained sentence-transformer models. The project processes large text files, segments them into sentences, cleans the sentences, and computes their relevance scores based on cosine similarity with the input query.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Processing Text](#processing-text)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
This project leverages pre-trained sentence-transformer models to identify and rank the most relevant sentences in a text document based on a given query. The process includes loading and cleaning the text, segmenting it into sentences, and calculating relevance scores using cosine similarity.

## Models
Default embedding models used:
- Qwen3-Embedding-4B → `Qwen/Qwen3-Embedding-4B`
- Jina-Embeddings-v3 → `jinaai/jina-embeddings-v3`
- Snowflake Arctic-Embed-L-v2 → `Snowflake/snowflake-arctic-embed-l-v2.0`
- Stella-en-1.5B-v5 → `NovaSearch/stella_en_1.5B_v5`
- GTE-Large → `thenlper/gte-large`
- EmbeddingGemma-300M → `google/embeddinggemma-300m`

The code loads models via Sentence-Transformers when available, and falls back to Hugging Face Transformers with mean pooling and L2 normalization otherwise. You can override the model list in the CLI with `--models`.

Backend selection
- By default, the loader prefers Sentence-Transformers if installed. To force HF Transformers (useful on serverless to avoid ST warnings and heavy deps), set `SSF_EMBED_BACKEND=hf` in your environment.

## Processing Text
The project includes several helper functions to process the text:
- **Text Chunking**: Splits the text into manageable chunks based on character count.
- **Sentence Segmentation**: Segments chunks into individual sentences using `pysbd`.
- **Cleaning Sentences**: Removes extra spaces and normalizes the text.

The `process` function calculates relevance scores for the sentences and prints the top-ranked sentences.

## Installation
Clone the repository:
```bash
git clone https://github.com/danielshort3/smart-sentence-finder.git
cd smart-sentence-finder
```

Install the required packages:
```base
pip install torch sentence-transformers pysbd
```

## Usage

Two options: CLI (recommended) or Docker.

### CLI

1. Install dependencies (torch + libs):
   ```bash
   pip install torch sentence-transformers pysbd tqdm transformers scikit-learn
   ```
2. Rank sentences against a query:
   ```bash
   python -m smart_sentence_finder.cli \
     rank \
     --file data/alice_in_wonderland.txt \
     --query "She wonders about things." \
     --top 5
   ```
   - Add `--models` to override the default list (space-separated).
   - Use `--chars-per-chunk` to control segmentation chunk size.

3. Benchmark models with silhouette score on Alice sentences:
   ```bash
   python -m smart_sentence_finder.cli \
     benchmark \
     --file data/alice_in_wonderland.txt \
     --max-sentences 1000 \
     --k-min 2 --k-max 10
   ```

### Docker

 Build and run using a recent PyTorch GPU image (default):

```bash
docker build -t smart-sentence-finder .

# Run rank (mount data, models, and output to persist downloads and results)
docker run --rm -it --gpus all \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/models" \
  -v "$PWD/output:/app/output" \
  -e HF_TOKEN=YOUR_HF_TOKEN \
  smart-sentence-finder \
  rank --file /app/data/alice_in_wonderland.txt --query "She wonders about things." --top 5 --output-dir /app/output

# Run benchmark
docker run --rm -it --gpus all \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/models" \
  -v "$PWD/output:/app/output" \
  -e HF_TOKEN=YOUR_HF_TOKEN \
  smart-sentence-finder \
  benchmark --file /app/data/alice_in_wonderland.txt --max-sentences 800 --output-dir /app/output
```

If you need CPU-only, change the `FROM` line in `Dockerfile` to a CPU tag (e.g., `pytorch/pytorch:2.8.0-cpu`) and rebuild. For GPU, run with:

```bash
docker run --rm -it --gpus all \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/models" \
  -v "$PWD/output:/app/output" \
  -e HF_TOKEN=YOUR_HF_TOKEN \
  smart-sentence-finder \
  --file /app/data/alice_in_wonderland.txt \
  --query "She wonders about things."
```

Model cache persistence
- The container caches models under `/models` using these env vars:
  - `HF_HOME=/models/huggingface`
  - `HUGGINGFACE_HUB_CACHE=/models/huggingface/hub`
  - `SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers`
- Mount a host directory (e.g. `-v "$PWD/models:/models"`) so downloads persist across runs and images.

Private/gated models
- Export a Hugging Face token: `export HF_TOKEN=...` (or `HUGGINGFACE_HUB_TOKEN`).
- Pass it to Docker with `-e HF_TOKEN=...` as shown above. Do not commit tokens to the repo.

### Serverless (AWS Lambda)

This repo includes a Lambda container image that bundles:
- CPU-only PyTorch, FastAPI + Mangum (ASGI on Lambda)
- The Snowflake embedding model cached in the image layer (no runtime download)
- Precomputed Alice-in-Wonderland sentences and embeddings

Lambda-specific notes
- The Lambda image uses HF Transformers by default (`SSF_EMBED_BACKEND=hf`) and omits `sentence-transformers` to keep cold start small and to avoid joblib multiprocess warnings.

Prereqs
- AWS CLI logged in; ECR access; Docker installed
- Optional: HF token if the model is gated: `export HF_TOKEN=...`

1) Precompute artifacts locally (optional if already present under `artifacts/`):
   ```bash
   python scripts/precompute_alice.py \
     --file data/alice_in_wonderland.txt \
     --model "Snowflake/snowflake-arctic-embed-l-v2.0" \
     --output-dir artifacts/snowflake_arctic_v2
   ```

2) Build and push the Lambda image (to ECR):
   ```bash
   # Configure as needed
   export AWS_PROFILE=default
   export AWS_REGION=us-east-2
   export ECR_REPO=smart-sentence-finder-lambda
   export IMAGE_TAG=latest
   export MODEL_NAME="Snowflake/snowflake-arctic-embed-l-v2.0"
   export HF_TOKEN=${HF_TOKEN:-}

   scripts/deploy/build_push_lambda.sh
   ```

3) Point your Lambda function to the new image (create if needed):
   ```bash
   ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --profile "$AWS_PROFILE")
   ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"

   # Create (first time)
   aws lambda create-function \
     --function-name smart-sentence-finder \
     --package-type Image \
     --code ImageUri="$ECR_URI" \
     --role arn:aws:iam::$ACCOUNT_ID:role/service-role/lambda-basic-exec \
     --region "$AWS_REGION" --profile "$AWS_PROFILE"

   # Or update existing
   aws lambda update-function-code \
     --function-name smart-sentence-finder \
     --image-uri "$ECR_URI" \
     --publish \
     --region "$AWS_REGION" --profile "$AWS_PROFILE"

   # Recommended: increase memory/time
   aws lambda update-function-configuration \
     --function-name smart-sentence-finder \
     --timeout 120 --memory-size 2048 \
     --region "$AWS_REGION" --profile "$AWS_PROFILE"
   ```

4) Enable a Function URL (simple HTTPS endpoint) and test:
   ```bash
   # Create once
   aws lambda create-function-url-config \
     --function-name smart-sentence-finder \
     --auth-type NONE \
     --cors "{\"AllowOrigins\":[\"*\"],\"AllowMethods\":[\"GET\",\"POST\"],\"AllowHeaders\":[\"*\"]}" \
     --region "$AWS_REGION" --profile "$AWS_PROFILE"

   URL=$(aws lambda get-function-url-config --function-name smart-sentence-finder --query FunctionUrl --output text --region "$AWS_REGION" --profile "$AWS_PROFILE")
   curl -sS "$URL/health"
   curl -sS -X POST "$URL/rank" -H 'content-type: application/json' -d '{"query":"She wonders about things.","top":5}'
   ```

Client integration (example)
```js
async function rank(query, top=5) {
  const url = "https://YOUR_FUNCTION_ID.lambda-url.us-east-2.on.aws/rank";
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, top })
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
```

Notes
- The image bundles the model and artifacts; runtime runs fully offline.
- To switch models, rebuild with `--build-arg MODEL_NAME=...` and repush.
- See `service/api.py` for the `/health` and `/rank` endpoints.

### Notebook (legacy)

The original Jupyter notebook `smart_sentence_finder.ipynb` remains in the repo, but the code has been refactored into Python modules under `src/smart_sentence_finder`. Prefer the CLI for running the workflow.

## Project Structure

```
src/
  smart_sentence_finder/
    __init__.py
    cli.py              # CLI entry point
    search.py           # Model loading and ranking
    text.py             # Chunking, segmentation, cleaning
data/
  alice_in_wonderland.txt (example file)
```
