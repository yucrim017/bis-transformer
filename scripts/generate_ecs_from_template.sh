#!/bin/bash

set -e

source aws.env

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-us-east-1}
PGPASSWORD=$(aws secretsmanager get-secret-value --secret-id postgres-password --query SecretString --output text)

echo "AWS Account ID: $AWS_ACCOUNT_ID"
echo "AWS Region: $AWS_REGION"
echo "PGPASSWORD: $PGPASSWORD"

export AWS_ACCOUNT_ID
export AWS_REGION
export PGPASSWORD

python scripts/generate_ecs_from_template.py \
    aws/ecs-mlflow-postgres-s3-task.template.json \
    aws/ecs-mlflow-postgres-s3-task.json

echo "Generated ECS task definition: aws/ecs-mlflow-postgres-s3-task.json"