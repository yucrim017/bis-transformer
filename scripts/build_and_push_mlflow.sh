#!/bin/bash

source .env

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

docker build -f docker/mlflow-aws.Dockerfile -t mlflow-aws .
docker tag mlflow-aws:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mlflow-aws:latest

aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mlflow-aws:latest