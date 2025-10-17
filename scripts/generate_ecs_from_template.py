import json
import os
import argparse
from dotenv import load_dotenv

def generate_ecs_from_template(template_path, output_path):
    # Load environment variables from aws.env
    load_dotenv("aws.env")
    
    with open(template_path, "r") as file:
        template = json.load(file)

    # Get AWS account ID
    aws_account_id = os.getenv("AWS_ACCOUNT_ID")
    if not aws_account_id:
        import subprocess
        aws_account_id = subprocess.check_output(
            ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"]
        ).decode().strip()

    # Update execution and task role ARNs
    template["executionRoleArn"] = f"arn:aws:iam::{aws_account_id}:role/ecsTaskExecutionRole"
    template["taskRoleArn"] = f"arn:aws:iam::{aws_account_id}:role/ecsTaskRole"

    # Update PostgreSQL container
    postgres_container = template["containerDefinitions"][0]
    postgres_container["environment"] = [
        {"name": "POSTGRES_USER", "value": os.getenv("PGUSER")},
        {"name": "POSTGRES_DB", "value": os.getenv("PGDB")},
        {"name": "S3_BACKUP_BUCKET", "value": os.getenv("S3_BUCKET")},
        {"name": "S3_BACKUP_KEY", "value": os.getenv("S3_BACKUP_KEY")},
    ]
    postgres_container["secrets"] = [
        {"name": "POSTGRES_PASSWORD", "valueFrom": f"arn:aws:secretsmanager:{os.getenv('AWS_REGION')}:{aws_account_id}:secret:postgres-password"}
    ]
    postgres_container["portMappings"][0]["containerPort"] = int(os.getenv("PGPORT", "5432"))

    # Update MLflow container
    mlflow_container = template["containerDefinitions"][1]
    mlflow_container["image"] = f"{aws_account_id}.dkr.ecr.{os.getenv('AWS_REGION')}.amazonaws.com/mlflow-aws:latest"
    mlflow_container["environment"] = [
        {"name": "MLFLOW_BACKEND_STORE_URI", "value": f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@localhost:{os.getenv('PGPORT')}/{os.getenv('PGDB')}"},
        {"name": "MLFLOW_DEFAULT_ARTIFACT_ROOT", "value": f"s3://{os.getenv('S3_BUCKET')}/{os.getenv('S3_ARTIFACTS_KEY')}"},
    ]
    mlflow_container["portMappings"][0]["containerPort"] = int(os.getenv("MLFLOW_PORT", "5000"))

    # Update log configurations
    for container_def in template["containerDefinitions"]:
        if "logConfiguration" in container_def:
            container_def["logConfiguration"]["options"]["awslogs-region"] = os.getenv("AWS_REGION")
            container_def["logConfiguration"]["options"]["awslogs-group"] = "/ecs/mlflow-postgres-s3"

    with open(output_path, "w") as file:
        json.dump(template, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("template_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    generate_ecs_from_template(args.template_path, args.output_path)