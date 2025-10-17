#!/bin/bash

set -e

source aws.env

echo "Starting cluster cleanup with PostgreSQL backup..."

# Backup PostgreSQL data before cleanup
echo "Creating PostgreSQL backup..."
if bash scripts/backup_postgres_s3.sh; then
    echo "PostgreSQL backup completed successfully"
else
    echo "Warning: PostgreSQL backup failed, but continuing with cleanup"
fi

# Delete all services
echo "Stopping and deleting services..."
aws ecs list-services --cluster bis-transformer --query 'serviceArns[*]' --output text | while read service; do
    if [ -n "$service" ] && [ "$service" != "None" ]; then
        service_name=$(basename "$service")
        echo "Stopping service: $service_name"
        # First, scale down the service to 0
        aws ecs update-service --cluster bis-transformer --service "$service_name" --desired-count 0 >/dev/null 2>&1
        echo "Waiting for service to stop..."
        sleep 10
        echo "Deleting service: $service_name"
        aws ecs delete-service --cluster bis-transformer --service "$service_name" --force >/dev/null 2>&1
    fi
done

# Delete security group
echo "Deleting security group..."
aws ec2 describe-security-groups --query 'SecurityGroups[0].GroupId' --output text | while read security_group; do
    if [ -n "$security_group" ] && [ "$security_group" != "None" ]; then
        echo "Deleting security group: $security_group"
        aws ec2 delete-security-group --group-id "$security_group" >/dev/null 2>&1
    fi
done

# Delete subnet
echo "Deleting subnet..."
aws ec2 describe-subnets --query 'Subnets[0].SubnetId' --output text | while read subnet; do
    if [ -n "$subnet" ] && [ "$subnet" != "None" ]; then
        echo "Deleting subnet: $subnet"
        aws ec2 delete-subnet --subnet-id "$subnet" >/dev/null 2>&1
    fi
done

# Delete VPC
echo "Deleting VPC..."
aws ec2 describe-vpcs --query 'Vpcs[0].VpcId' --output text | while read vpc; do
    if [ -n "$vpc" ] && [ "$vpc" != "None" ]; then
        echo "Deleting VPC: $vpc"
        aws ec2 delete-vpc --vpc-id "$vpc" >/dev/null 2>&1
    fi
done

# Delete task definitions
echo "Deleting task definitions..."
aws ecs list-task-definitions --family-prefix mlflow-postgres-s3 --query 'taskDefinitionArns[*]' --output text | while read task_def; do
    if [ -n "$task_def" ] && [ "$task_def" != "None" ]; then
        echo "Deleting task definition: $task_def"
        aws ecs deregister-task-definition --task-definition "$task_def" >/dev/null 2>&1
    fi
done

# Delete cluster
echo "Deleting cluster: bis-transformer"
aws ecs delete-cluster --cluster bis-transformer >/dev/null

# Delete confirmation
sleep 5
aws ecs describe-clusters --clusters bis-transformer >/dev/null
if [ $? -eq 0 ]; then
    echo "Cluster is remaining. Please check the cluster manually."
    exit 1
else
    echo "Cluster successfully deleted"
    exit 0
fi