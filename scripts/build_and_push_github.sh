#!/bin/bash

# GitHub Container Registryの設定
GITHUB_USERNAME="yucrim017"
GITHUB_REPOSITORY="mlflow-aws"
GITHUB_TAG="latest"

echo "Building Docker image for GitHub Container Registry..."
echo "Repository: ghcr.io/${GITHUB_USERNAME}/${GITHUB_REPOSITORY}:${GITHUB_TAG}"

# Dockerfileのパス
DOCKERFILE_PATH="../docker/mlflow-aws.Dockerfile"

# Dockerイメージをビルド
echo "Building Docker image..."
docker build -f ${DOCKERFILE_PATH} -t ghcr.io/${GITHUB_USERNAME}/${GITHUB_REPOSITORY}:${GITHUB_TAG} ../

# GitHub Container Registryにプッシュ
echo "Pushing to GitHub Container Registry..."
docker push ghcr.io/${GITHUB_USERNAME}/${GITHUB_REPOSITORY}:${GITHUB_TAG}

echo "Docker image pushed successfully to GitHub Container Registry!"
echo "Image: ghcr.io/${GITHUB_USERNAME}/${GITHUB_REPOSITORY}:${GITHUB_TAG}"
