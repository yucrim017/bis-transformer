variable "aws_region" {
    description = "AWS region"
    type = string
}

variable "aws_account_id" {
    description = "AWS account ID"
    type = string
}

variable "project_name" {
    description = "Project name"
    type = string
    default = "bis-transformer"
}

variable "ecs_task_cpu" {
    description = "ECS task CPU"
    type = number
    default = 1024
}

variable "ecs_task_memory" {
    description = "ECS task memory"
    type = number
    default = 2048
}

variable "mlflow_port" {
    description = "MLflow server port"
    type = number
    default = 5000
}

variable "pguser" {
    description = "PostgreSQL user"
    type = string
}

variable "pgpassword" {
    description = "PostgreSQL password"
    type = string
}

variable "pghost" {
    description = "PostgreSQL host"
    type = string
}

variable "pgport" {
    description = "PostgreSQL port"
    type = number
}

variable "pgdb" {
    description = "PostgreSQL database name"
    type = string
}

variable "s3_bucket" {
    description = "S3 bucket name"
    type = string
}

variable "s3_artifacts_key" {
    description = "S3 artifacts key"
    type = string
}

variable "ecs_task_execution_role_name" {
    description = "Existing ECS task execution role name"
    type = string
    default = "ecsTaskExecutionRole"
}

variable "ecs_task_role_name" {
    description = "Existing ECS task role name"
    type = string
    default = "ecsTaskRole"
}