#!/bin/bash

set -e

source aws.env

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-us-east-1}

echo "AWS Account ID: $AWS_ACCOUNT_ID"
echo "AWS Region: $AWS_REGION"

export AWS_ACCOUNT_ID
export AWS_REGION

python scripts/launch_from_config.py \
    --config config.yaml
    --