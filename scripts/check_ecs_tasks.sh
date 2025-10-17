#!/bin/bash

echo "Checking ECS tasks..."
TASK_ARN=$(aws ecs list-tasks --cluster bis-transformer --service-name bis-mlflow-postgres --query 'taskArns[0]' --output text)
if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" = "None" ]; then
    echo "Task not found"
    exit 1
fi

echo "Task ARN: $TASK_ARN"
echo ""
aws ecs describe-tasks \
    --cluster bis-transformer \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].{LastStatus:lastStatus,DesiredStatus:desiredStatus,HealthStatus:healthStatus,Containers:containers[].{Name:name,LastStatus:lastStatus}}' || echo "Task not found"

echo "Getting security group port settings..."
aws ec2 describe-security-groups \
    --group-ids $SECURITY_GROUP_ID \
    --query 'SecurityGroups[0].IpPermissions[?FromPort==`5000`]'
echo ""
echo "Changing security group port settings..."
SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=bis-transformer-sg" \
    --query "SecurityGroups[0].GroupId" \
    --output text) || echo "Security group not found"

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 5432 \
    --cidr 0.0.0.0/0 || echo "Port 5432 rule already exists"

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 5000 \
    --cidr 0.0.0.0/0 || echo "Port 5000 rule already exists"

echo -e "\nGetting public IP address..."
NETWORK_INTERFACE_ID=$(aws ecs describe-tasks \
    --cluster bis-transformer \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
    --output text) || echo "Network interface not found"

PUBLIC_IP=$(aws ec2 describe-network-interfaces \
    --network-interface-ids "$NETWORK_INTERFACE_ID" \
    --query 'NetworkInterfaces[0].Association.PublicIp' \
    --output text) || echo "Public IP not found"

echo "Public IP: $PUBLIC_IP"
echo "MLflow URL: http://$PUBLIC_IP:5000"