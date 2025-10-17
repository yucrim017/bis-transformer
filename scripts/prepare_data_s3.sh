#!/bin/bash

set -e

source aws.env

AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)

# Get MLflow server public IP from AWS ECS
echo "Getting MLflow server public IP..."
TASK_ARN=$(aws ecs list-tasks --cluster bis-transformer --query 'taskArns[0]' --output text)

if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
    NETWORK_INTERFACE_ID=$(aws ecs describe-tasks \
        --cluster bis-transformer \
        --tasks "$TASK_ARN" \
        --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
        --output text)
    
    PUBLIC_IP=$(aws ec2 describe-network-interfaces \
        --network-interface-ids "$NETWORK_INTERFACE_ID" \
        --query 'NetworkInterfaces[0].Association.PublicIp' \
        --output text 2>/dev/null)
    
    if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "None" ]; then
        MLFLOW_TRACKING_URI="http://${PUBLIC_IP}:5000"
        echo "MLflow server found at: $MLFLOW_TRACKING_URI"
    else
        echo "Warning: Could not get public IP. Using default tracking URI."
    fi
else
    echo "Warning: No ECS task found. Using default tracking URI."
fi

export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export MLFLOW_TRACKING_URI
export S3_BUCKET
export S3_ARTIFACTS_KEY
export AWS_REGION

python scripts/prepare_with_mlflow.py

python scripts/upload_data_s3.py \
    --bucket-name $S3_BUCKET \
    --region-name $AWS_REGION \
    --aws-access-key-id $AWS_ACCESS_KEY_ID \
    --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
    --data-dir data/processed