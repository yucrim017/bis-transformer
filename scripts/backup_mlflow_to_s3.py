import boto3
import subprocess
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('aws.env')

def dump_postgres_from_ecs():
    """
    Dump postgres mlflow database from ECS
    """
    ecs = boto3.client('ecs', region_name=os.getenv('AWS_REGION'))

    # get the task definition
    tasks = ecs.list_tasks(
        cluster=os.getenv('ECS_CLUSTER'),
        serviceName=os.getenv('ECS_SERVICE'),
        desiredStatus='RUNNING'
    )

    if not tasks['taskArns']:
        print("No running tasks found")
        return None

    task_arn = tasks['taskArns'][0]
    task_id = task_arn.split('/')[-1]

    print(f"Dumping postgres data from ECS task: {task_id}")

    dump_command = (
        f"aws ecs execute-command"
        f"--cluster {os.getenv('ECS_CLUSTER')}"
        f"--task {task_id}"
        f"--container {os.getenv('ECS_POSTGRES_CONTAINER')}"
        f"--interactive "
        f"--command '/bin/bash -c \"pg_dump -U mlflow mlflow > /tmp/{os.getenv('S3_BACKUP_KEY')}\"'"
    )

    subprocess.run(dump_command, shell=True, check=True)

    # get the dump file from the container
    print(f"Connecting to Postgres container ...")

    # get ALB endpoint
    endpoint = os.getenv("MLFLOW_POSTGRES_ENDPOINT")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dump_file = f"/tmp/mlflow_aws_{timestamp}.sql"

    env = os.environ.copy()
    env["PGPASSWORD"] = os.getenv("PGPASSWORD")

    # download the dump file from the container
    result = subprocess.run([
        'pg_dump',
        '-h', endpoint,
        '-U', 'mlflow',
        '-d', 'mlflow',
        '-F', 'p', # format: plain
        '-f', dump_file,
        '--clean',
        '--if-exists',
    ], env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error dumping postgres data: {result.stderr}")
        raise Exception("Dump failed")

    size_mb = os.path.getsize(dump_file) / (1024 * 1024)
    print(f"  Dump completed: {dump_file} ({size_mb:.2f} MB)")

    return dump_file

def upload_to_s3(dump_file):
    """
    Upload the dump file to S3
    """
    s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
    bucket = os.getenv('S3_BUCKET') + "-backups"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    key_latest = 'mlflow_latest.sql'
    key_timestamp = f'history/mlflow_{timestamp}.sql'
    
    print(f"Uploading to S3 ...")

    s3.upload_file(dump_file, bucket, key_latest)
    print(f"  Uploaded: {bucket}/{key_latest}")

    s3.upload_file(dump_file, bucket, key_timestamp)
    print(f"  Uploaded: {bucket}/{key_timestamp}")

    # remove the local dump file
    os.remove(dump_file)

    return f"s3://{bucket}/{key_latest}"

def download_artifacts_from_s3():
    """
    Download artifacts from S3
    """
    s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
    bucket = os.getenv('S3_BUCKET') + "-artifacts"
    local_dir = 'mlflow/artifacts'

    print(f"Downloading artifacts from S3 ...")

    os.makedirs(local_dir, exist_ok=True)

    pagenator = s3.get_paginator('list_objects_v2')
    pages = pagenator.paginate(Bucket=bucket)

    file_count = 0
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            local_file = os.path.join(local_dir, key.split('/')[-1])
            os.makedirs(os.path.dirname(local_file), exist_ok=True)

            s3.download_file(bucket, key, local_file)
            file_count += 1

            if file_count % 10 == 0:
                print(f"  Downloading ... {file_count} files")

    print(f"  Downloaded {file_count} files from S3")
    print(f"  Local directory: {local_dir}")

def main():
    print("=" * 80)
    print("Backup MLflow data to S3")
    print("=" * 80)

    # --- 1. create Postgres dump ---
    dump_file = dump_postgres_from_ecs()

    # --- 2. upload dump file to S3 ---
    s3_uri = upload_to_s3(dump_file)

    # --- 3. download artifacts from S3 ---
    response = input(f"Download artifacts from S3? (y/n): ")
    if response.lower() == 'y':
        download_artifacts_from_s3()
    
    print("\nBackup completed.")
    print(f"\nPostgres dump: {s3_uri}")
    print("\nNext steps:")
    print("  Check the Postgres dump uploaded to S3")
    print(f"     aws s3 ls {s3_uri} --recursive")
    print("  Restore the Postgres dump to the local database")
    print(f"     python scripts/restore_to_local_mlflow.py")

if __name__ == "__main__":
    main()