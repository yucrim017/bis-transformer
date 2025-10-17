#!/bin/bash

set -e

source aws.env

echo "Start ECS MLflow infrastructure..."
echo ""

# Build and push mlflow-aws image
echo "Build and push mlflow-aws image..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=${AWS_REGION:-us-east-1}

echo "  AWS Account ID: $AWS_ACCOUNT_ID"
echo "  AWS Region: $AWS_REGION"
echo ""

export AWS_ACCOUNT_ID
export AWS_REGION
REPO_NAME=mlflow-aws

echo "Build and push mlflow-aws image..."
echo ""
echo "Checking if ECR repository exists..."

if ! aws ecr describe-repositories \
    --repository-names $REPO_NAME \
    --region $AWS_REGION >/dev/null 2>&1; then
    echo "Creating ECR repository $REPO_NAME..."
    aws ecr create-repository --repository-name $REPO_NAME --region $AWS_REGION >/dev/null
    if [ $? -eq 0 ]; then
        echo "ECR repository $REPO_NAME created"
    else
        echo "Failed to create ECR repository $REPO_NAME"
        exit 1
    fi
else
    echo "ECR repository $REPO_NAME already exists"
fi

echo ""

aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "Building $REPO_NAME image..."
echo ""

docker build --platform linux/amd64 --load -t ${REPO_NAME} -f docker/mlflow-aws.Dockerfile .
docker tag ${REPO_NAME}:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest

echo "Pushing $REPO_NAME image to ECR..."
echo ""
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest

echo "Successfully built and pushed $REPO_NAME image to ECR."
echo ""

echo "Starting ECS mlflow server ..."
echo ""

# create log group
echo "Creating log group ..."
aws logs list-log-groups \
    --log-group-name /ecs/mlflow-postgres-s3 \
    --query 'logGroups[].logGroupName' \
    --output text | grep -q "/ecs/mlflow-postgres-s3"
if [ $? -eq 0 ]; then
    echo "Log group already exists: /ecs/mlflow-postgres-s3"
else
    echo "Log group does not exist"
    aws logs create-log-group \
        --log-group-name /ecs/mlflow-postgres-s3 \
        --output text > /dev/null
    if [ $? -eq 0 ]; then
        echo "Log group created: /ecs/mlflow-postgres-s3"
    else
        echo "$?"
        echo "Failed to create log group: /ecs/mlflow-postgres-s3"
        exit 1
    fi
fi

echo ""
echo "Creating cluster ..."
if aws ecs describe-clusters \
    --clusters "bis-transformer" \
    --query 'clusters[0].status' \
    --output text | grep -q "INACTIVE"; then
    aws ecs create-cluster \
        --cluster-name "bis-transformer" \
        --capacity-providers FARGATE \
        --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1\
        --output text >/dev/null
    echo "Cluster created: bis-transformer"
    sleep 5
elif aws ecs describe-clusters \
    --clusters "bis-transformer" \
    --query 'clusters[0].status' \
    --output text | grep -q "ACTIVE"; then
    echo "Active cluster exists: bis-transformer"
else
    aws ecs create-cluster \
        --cluster-name "bis-transformer" \
        --capacity-providers FARGATE \
        --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1\
        --output text >/dev/null
    echo "Cluster created: bis-transformer"
fi

echo ""

# check and register task definition
echo "Checking task definition..."
EXISTING_TASK_DEFS=$(aws ecs list-task-definitions \
    --family-prefix "mlflow-postgres-s3" \
    --query 'taskDefinitionArns[0]' \
    --output text 2>/dev/null)

if [ -n "$EXISTING_TASK_DEFS" ] && [ "$EXISTING_TASK_DEFS" != "None" ]; then
    echo "Existing task definition found: $EXISTING_TASK_DEFS"
    TASK_DEF_ARN="$EXISTING_TASK_DEFS"
else
    echo "Registering new task definition ..."
    echo "Using file: aws/ecs-mlflow-postgres-s3-task.json"
    
    TASK_DEF_ARN=$(aws ecs register-task-definition \
        --cli-input-json file://aws/ecs-mlflow-postgres-s3-task.json \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)

    if [ $? -eq 0 ] && [ -n "$TASK_DEF_ARN" ]; then
        echo "New task definition registered: $TASK_DEF_ARN"
    else
        echo "Failed to register new task definition"
        exit 1
    fi
fi

echo ""

# get subnets and security group
SUBNET_IDS=$(aws ec2 describe-subnets \
    --filters "Name=default-for-az,Values=true" \
    --query 'Subnets[*].SubnetId' \
    --output text | tr '\t' ',')

SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=bis-transformer-sg" \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

echo "Using subnets: $SUBNET_IDS"
echo ""
echo "Using security group: $SECURITY_GROUP_ID"
echo ""

# create/update ECS service
echo "Checking if service exists..."
SERVICE_EXISTS=$(aws ecs describe-services \
    --cluster "bis-transformer" \
    --services "mlflow-postgres-s3" \
    --query 'services[0].serviceName' \
    --output text 2>/dev/null)

if [ -n "$SERVICE_EXISTS" ] && [ "$SERVICE_EXISTS" != "None" ]; then
    echo "Updating existing service: $SERVICE_EXISTS"
    aws ecs update-service \
        --cluster "bis-transformer" \
        --service "mlflow-postgres-s3" \
        --task-definition "$TASK_DEF_ARN" \
        --force-new-deployment >/dev/null
    if [ $? -eq 0 ]; then
        echo "Service updated"
    else
        echo "Failed to update service"
        exit 1
    fi
else
    echo "Creating new service ..."
    SERVICE_NAME=$(aws ecs create-service \
        --cluster "bis-transformer" \
        --service-name "mlflow-postgres-s3" \
        --task-definition "$TASK_DEF_ARN" \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={
            subnets=[$SUBNET_IDS],
            securityGroups=[$SECURITY_GROUP_ID],
            assignPublicIp=ENABLED
        }" \
        --query 'service.serviceName' \
        --output text 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "Service created: $SERVICE_NAME"
    else
        echo "Failed to create service"
        exit 1
    fi
fi

echo ""

echo "Waiting for service to be ready ..."
while true; do
    aws ecs describe-services \
        --cluster "bis-transformer" \
        --services "mlflow-postgres-s3" \
        --query 'services[0].status' \
        --output text | grep -q "ACTIVE" || true
    if [ $? -eq 0 ]; then
        break
    fi
    sleep 1
done

echo "Service is active: mlflow-postgres-s3"

echo ""
echo "Waiting for task to be ready (2-3 minutes) ..."
while true; do
    TASK_ARN=$(aws ecs list-tasks \
        --cluster "bis-transformer" \
        --service-name "mlflow-postgres-s3" \
        --query 'taskArns[0]' \
        --output text) || true
    if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ] && [ "$TASK_ARN" != "null" ]; then
        break
    fi
    sleep 1
done

echo "Task ARN: ${TASK_ARN}"
echo ""
echo "Waiting for task to be running ..."

while true; do
    aws ecs describe-tasks \
        --cluster "bis-transformer" \
        --tasks "$TASK_ARN" \
        --query 'tasks[0].lastStatus' \
        --output text | grep -q "RUNNING" || true
    if [ $? -eq 0 ]; then
        break
    fi
    sleep 1
done

echo "Task is running"
echo ""

# Get public IP address
echo "Authorizing security group ingress ..."

SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=bis-transformer-sg" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null)

echo "SECURITY_GROUP_ID: $SECURITY_GROUP_ID"

if [ -z "$SECURITY_GROUP_ID" ] || [ "$SECURITY_GROUP_ID" = "None" ]; then
    echo "Creating security group..."
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name bis-transformer-sg \
        --description "Security group for bis-transformer ECS tasks" \
        --query 'GroupId' \
        --output text)
    echo "Created security group: $SECURITY_GROUP_ID"
fi

echo ""

echo "Adding security group rules..."
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 5432 \
    --cidr 0.0.0.0/0 >/dev/null 2>&1 || echo "Port 5432 rule already exists"

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 5000 \
    --cidr 0.0.0.0/0 >/dev/null 2>&1 || echo "Port 5000 rule already exists"

echo "Security group rules configured"

echo ""
echo "Getting public IP address..."

if [ -n "$TASK_ARN" ]; then
    NETWORK_INTERFACE_ID=$(aws ecs describe-tasks \
    --cluster bis-transformer \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
    --output text)

    echo "NETWORK_INTERFACE_ID: $NETWORK_INTERFACE_ID"

    # Get public IP from network interface
    PUBLIC_IP=$(aws ec2 describe-network-interfaces \
        --network-interface-ids "$NETWORK_INTERFACE_ID" \
        --query 'NetworkInterfaces[0].Association.PublicIp' \
        --output text 2>/dev/null)

    echo "PUBLIC_IP: $PUBLIC_IP"
    echo ""

    if [ -n "$PUBLIC_IP" ]; then
        echo "MLflow UI is ready at: http://${PUBLIC_IP}:5000"
    else
        echo "Public IP not available yet. Check AWS console for more details."
    fi
else
    echo "No task found. Check service status."
fi

echo ""
echo "To check logs:"
echo "　　aws logs tail /ecs/mlflow-postgres-s3 --follow"