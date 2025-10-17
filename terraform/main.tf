terraform {
    required_version = ">=1.0"
    required_providers {
        aws = {
            source = "hashicorp/aws",
            version = "~> 5.0"
        }
    }
}

provider "aws" {
    region = var.aws_region
}

data "aws_vpc" "default" {
    default = true
}

data "aws_subnets" "default" {
    filter {
        name = "vpc-id"
        values = [data.aws_vpc.default.id]
    }
}

# Use existing S3 bucket
data "aws_s3_bucket" "mlflow_artifacts" {
    bucket = var.s3_bucket
}

# Use existing IAM roles
data "aws_iam_role" "ecs_task_execution_role" {
    name = var.ecs_task_execution_role_name
}

data "aws_iam_role" "ecs_task_role" {
    name = var.ecs_task_role_name
}

# Attach S3 access policy to existing ECS task role
resource "aws_iam_role_policy_attachment" "ecs_task_role_s3_policy" {
    role = data.aws_iam_role.ecs_task_role.name
    policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_ecs_cluster" "mlflow" {
    name = "mlflow-cluster"
    
    setting {
        name = "containerInsights"
        value = "enabled"
    }
    tags = {
        Name = "MLflow ECS Cluster"
    }
}

# ECS task definition
resource "aws_ecs_task_definition" "mlflow" {
    family = "mlflow"
    network_mode = "awsvpc"
    requires_compatibilities = ["FARGATE"]
    cpu = var.ecs_task_cpu
    memory = var.ecs_task_memory
    execution_role_arn = data.aws_iam_role.ecs_task_execution_role.arn
    task_role_arn = data.aws_iam_role.ecs_task_role.arn

    container_definitions = jsonencode([
        {
            name = "mlflow"
            image = "${var.aws_account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/mlflow-aws:latest"
            essential = true
            portMappings = [
                {
                    containerPort = var.mlflow_port
                    protocol = "tcp"
                }
            ]
            environment = [
                {
                    name = "MLFLOW_DEFAULT_ARTIFACT_ROOT"
                    value = "s3://${data.aws_s3_bucket.mlflow_artifacts.bucket}/${var.s3_artifacts_key}"
                },
                {
                    name = "MLFLOW_BACKEND_STORE_URI"
                    value = "postgresql://${var.pguser}:${var.pgpassword}@${var.pghost}:${var.pgport}/${var.pgdb}"
                },
                {
                    name = "AWS_DEFAULT_REGION"
                    value = var.aws_region
                },
                {
                    name = "PGUSER"
                    value = var.pguser
                },
                {
                    name = "PGHOST"
                    value = var.pghost
                },
                {
                    name = "PGPORT"
                    value = tostring(var.pgport)
                },
                {
                    name = "PGDB"
                    value = var.pgdb
                }
            ]
            secrets = [
                {
                    name = "PGPASSWORD"
                    valueFrom = var.pgpassword
                }
            ]
            logConfiguration = {
                logDriver = "awslogs"
                options = {
                    awslogs-group = "/ecs/mlflow"
                    awslogs-region = var.aws_region
                    awslogs-stream-prefix = "mlflow"
                }
            }

            healthCheck = {
                command = ["CMD-SHELL", "curl -f http://localhost:${var.mlflow_port}/health || exit 1"]
                interval = 30
                timeout = 5
                retries = 3
                startPeriod = 30
            }
        }
    ])
    tags = {
        Name = "MLflow ECS Task Definition"
    }
}

# ECS service
resource "aws_ecs_service" "mlflow" {
    name = "mlflow-service"
    cluster = aws_ecs_cluster.mlflow.id
    task_definition = aws_ecs_task_definition.mlflow.arn
    desired_count = 1
    launch_type = "FARGATE"

    network_configuration {
        subnets = data.aws_subnets.default.ids
        assign_public_ip = true
        security_groups = [aws_security_group.mlflow.id]
    }

    tags = {
        Name = "MLflow ECS Service"
    }
}

# Security Groups
resource "aws_security_group" "mlflow" {
    name_prefix = "mlflow-"
    vpc_id = data.aws_vpc.default.id

    ingress {
        from_port = var.mlflow_port
        to_port = var.mlflow_port
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    egress {
        from_port = 0
        to_port = 0
        protocol = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }

    tags = {
        Name = "MLflow Security Group"
    }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "mlflow" {
    name = "/ecs/mlflow"
    retention_in_days = 30
    tags = {
        Name = "MLflow CloudWatch Log Group"
    }
}