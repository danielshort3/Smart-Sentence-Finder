#!/usr/bin/env bash
set -euo pipefail

# Build and push the Lambda container with bundled model + artifacts

# Config (override via env)
: "${AWS_PROFILE:=default}"
: "${AWS_REGION:=us-east-2}"
: "${ECR_REPO:=smart-sentence-finder-lambda}"
: "${IMAGE_TAG:=latest}"

# Model/token (optional)
: "${MODEL_NAME:=Snowflake/snowflake-arctic-embed-l-v2.0}"
: "${HF_TOKEN:=}"
: "${FORCE_REEMBED:=}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --profile "$AWS_PROFILE")
ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"

echo "Building Docker image with model '$MODEL_NAME' -> $ECR_URI"

docker build -f Dockerfile.lambda \
  --build-arg MODEL_NAME="$MODEL_NAME" \
  --build-arg HF_TOKEN="$HF_TOKEN" \
  ${FORCE_REEMBED:+--build-arg FORCE_REEMBED="$FORCE_REEMBED"} \
  -t "$ECR_REPO:$IMAGE_TAG" .

aws ecr describe-repositories --repository-name "$ECR_REPO" --region "$AWS_REGION" --profile "$AWS_PROFILE" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$ECR_REPO" --image-scanning-configuration scanOnPush=true --region "$AWS_REGION" --profile "$AWS_PROFILE" >/dev/null

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker tag "$ECR_REPO:$IMAGE_TAG" "$ECR_URI"
docker push "$ECR_URI"

echo "Pushed: $ECR_URI"

cat <<'TIPS'
Next steps:
1) Update Lambda code to this image:
   aws lambda update-function-code \
     --function-name smart-sentence-finder \
     --image-uri "$ECR_URI" \
     --publish \
     --region "$AWS_REGION" --profile "$AWS_PROFILE"

2) (Optional) Move alias:
   VER=\$(aws lambda publish-version --function-name smart-sentence-finder --query Version --output text --region "$AWS_REGION" --profile "$AWS_PROFILE")
   aws lambda update-alias --function-name smart-sentence-finder --name prod --function-version "$VER" --region "$AWS_REGION" --profile "$AWS_PROFILE"

3) Function URL test:
   URL=\$(aws lambda get-function-url-config --function-name smart-sentence-finder --query FunctionUrl --output text --region "$AWS_REGION" --profile "$AWS_PROFILE")
   curl -sS "$URL/health" | jq .
   curl -sS -X POST "$URL/rank" -H 'content-type: application/json' -d '{"query":"She wonders about things.","top":5}' | jq .
TIPS
