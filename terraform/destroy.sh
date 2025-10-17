#!/bin/bash

source ../.env

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
AWS_REGION=${AWS_DEFAULT_REGION}

echo "AWS_ACCOUNT_ID: ${AWS_ACCOUNT_ID}"
echo "AWS_REGION: ${AWS_REGION}"

terraform destroy -auto-approve \
    -var "aws_account_id=${AWS_ACCOUNT_ID}" \
    -var "aws_region=${AWS_REGION}" \
    -var "ecs_task_cpu=${ECS_TASK_CPU}" \
    -var "ecs_task_memory=${ECS_TASK_MEMORY}" \
    -var "mlflow_port=${MLFLOW_PORT}" \
    -var "pguser=${PGUSER}" \
    -var "pgpassword=${PGPASSWORD}" \
    -var "pghost=${PGHOST}" \
    -var "pgport=${PGPORT}" \
    -var "pgdb=${PGDB}" \
    -var "s3_bucket=${S3_BUCKET}" \
    -var "s3_artifacts_key=${S3_ARTIFACTS_KEY}"