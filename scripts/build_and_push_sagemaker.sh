#!/bin/bash

# Usage: build_and_push_sagemaker.sh [region] [account-id] [repository-name] [tag]
set -e

REGION=${1:-us-east-1}
ACCOUNT_ID=${2:-$(aws sts get-caller-identity --query Account --output text)}
REPOSITORY_NAME=${3:-bis-transformer}
TAG=${4:-latest}

echo ""
echo "Region: $REGION"
echo "Account ID: $ACCOUNT_ID"
echo "Repository: $REPOSITORY_NAME"
echo "Tag: $TAG"
echo ""

# ECR URI
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:${TAG}"
echo "Image URI: $ECR_URI"
echo ""

# Login to ECR
echo "[1/4] Logging in to ECR..."
aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Check if ECR repository exists
echo "[2/4] Checking ECR repository..."
if ! aws ecr describe-repositories --repository-names ${REPOSITORY_NAME} --region ${REGION} > /dev/null 2>&1; then
    echo "Creating ECR repository: ${REPOSITORY_NAME}"
    aws ecr create-repository --repository-name ${REPOSITORY_NAME} --region ${REGION}
else
    echo "ECR repository already exists: ${REPOSITORY_NAME}"
fi

echo ""
echo "[3/4] Building Docker image..."
# Move to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Build context: $PROJECT_ROOT"

# Build Docker image
docker build \
    -f docker/sagemaker.Dockerfile \
    -t ${REPOSITORY_NAME}:${TAG} \
    .
docker tag ${REPOSITORY_NAME}:${TAG} ${ECR_URI}

echo "[4/4] Pushing image to ECR..."
docker push ${ECR_URI}

echo "Successfully pushed image to ECR: ${ECR_URI}"
