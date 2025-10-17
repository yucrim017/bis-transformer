#!/bin/bash

set -e

source aws.env

echo "Starting PostgreSQL backup to S3..."

# Get ECS task details
TASK_ARN=$(aws ecs list-tasks --cluster bis-transformer --query 'taskArns[0]' --output text)

if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" = "None" ]; then
    echo "Error: No ECS task found"
    exit 1
fi

echo "Found task: $TASK_ARN"

# Get task details
TASK_DETAILS=$(aws ecs describe-tasks --cluster bis-transformer --tasks "$TASK_ARN")

# Check if PostgreSQL container is running
POSTGRES_STATUS=$(aws ecs describe-tasks --cluster bis-transformer --tasks "$TASK_ARN" \
    --query 'tasks[0].containers[?name==`postgres-backend`].lastStatus' --output text)

if [ "$POSTGRES_STATUS" != "RUNNING" ]; then
    echo "Error: PostgreSQL container is not running (status: $POSTGRES_STATUS)"
    exit 1
fi

echo "PostgreSQL container is running"

# Create backup directory
BACKUP_DIR="backups/postgres/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Get PostgreSQL connection details
PGHOST=$(aws ecs describe-tasks --cluster bis-transformer --tasks "$TASK_ARN" \
    --query 'tasks[0].containers[?name==`postgres-backend`].networkInterfaces[0].privateIpv4Address' --output text)
PGPORT=$PGPORT
PGUSER=$PGUSER
PGDB=$PGDB

echo "Connecting to PostgreSQL at $PGHOST:$PGPORT"

# Set password
export PGPASSWORD=$(aws secretsmanager get-secret-value --secret-id postgres-password --query SecretString --output text)

# Create database dump
echo "Creating database dump..."
pg_dump -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
    --format=custom \
    --verbose \
    --file="$BACKUP_DIR/mlflow_backup.dump"

# Create SQL dump as well
echo "Creating SQL dump..."
pg_dump -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDB" \
    --format=plain \
    --verbose \
    --file="$BACKUP_DIR/mlflow_backup.sql"

# Create backup metadata
cat > "$BACKUP_DIR/backup_info.json" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "database": "$PGDB",
    "host": "$PGHOST",
    "port": "$PGPORT",
    "user": "$PGUSER",
    "backup_type": "full",
    "files": [
        "mlflow_backup.dump",
        "mlflow_backup.sql"
    ]
}
EOF

echo "Backup completed successfully!"
echo "Backup location: $BACKUP_DIR"

# # Upload to S3
# echo "Uploading backup to S3..."
# aws s3 sync "$BACKUP_DIR" "s3://$S3_BUCKET/$S3_BACKUP_KEY/$(date +%Y%m%d_%H%M%S)/" \
#     --region "$AWS_REGION"

# echo "Backup completed successfully!"
# echo "Backup location: s3://$S3_BUCKET/$S3_BACKUP_KEY/$(date +%Y%m%d_%H%M%S)/"

# # Clean up local backup files
# rm -rf "$BACKUP_DIR"

# echo "Local backup files cleaned up"
