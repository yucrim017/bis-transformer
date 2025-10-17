from dotenv import load_dotenv
import boto3
import subprocess
import os
import time
import docker
from pathlib import Path

from bistransformer.storage import S3Manager

load_dotenv()

def ensure_local_mlflow_running():
    """
    Start the local MLflow server if it's not running
    """
    print("Starting local MLflow server ...")

    # start container using docker-compose
    result = subprocess.run(
        ['docker-compose', 'up', '-d'],
        cwd='docker',
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error starting local MLflow server: {result.stderr}")
        raise Exception("Failed to start local MLflow server")

    print("Waiting for service to be ready ...")
    time.sleep(10)

    # wait for postgres to be ready
    for i in range(30):
        result = subprocess.run([
            'docker-compose', 
            'exec', 
            '-T', 'postgres', 
            'pg_isready',
            '-U', 'mlflow',
            '-d', 'mlflow'
        ], capture_output=True)
        if result.returncode == 0:
            print("Postgres is ready")
            break
        time.sleep(1)
    else:
        raise Exception("Timeout waiting for starting Postgres")

def download_from_s3():
    """
    Download the Postgres dump from S3
    """
    s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
    bucket = os.getenv('S3_BUCKET') + "-backups"
    key = 'mlflow_latest.sql'
    local_file = f'mlflow/{os.getenv("S3_BACKUP_KEY")}'

    print("Downloading Postgres dump from S3: s3://{bucket}/{key} -> {local_file}")

    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    s3.download_file(bucket, key, local_file)
    
    size_mb = os.path.getsize(local_file) / (1024 * 1024)
    print(f"  Download completed: {local_file} ({size_mb:.2f} MB)")

    return local_file
    
def restore_to_local_postgres(dump_file):
    """
    Restore the Postgres database from the dump file
    """
    print("Restoring Postgres database from the dump file ...")

    # connect to local postgres container
    result = subprocess.run([
        'docker-compose', 
        'exec', 
        '-T', 'postgres', 
        'psql',
        '-U', 'mlflow', 
        '-d', 'mlflow', '-f'
    ], stdin=open(dump_file, 'r'), capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error restoring Postgres database: {result.stderr}")
        raise Exception("Failed to restore Postgres database")
    
    print("Postgres database restored.")

def sync_artifacts_to_minio():
    """
    Sync the S3 artifacts to Minio
    """
    print("Syncing artifacts to Minio ...")

    artifacts_dir = Path('mlflow/artifacts')

    if not artifacts_dir.exists():
        print(f"  Artifacts directory not found: {artifacts_dir}")
        print("  Please run the backup script first")
        return
    
    # minio s3 client
    s3_minio = S3Manager(
        bucket_name=os.getenv('MINIO_BUCKET'),
        access_key_id=os.getenv('MINIO_ACCESS_KEY'),
        secret_access_key=os.getenv('MINIO_SECRET_KEY'),
        endpoint_url=os.getenv('MINIO_ENDPOINT')
    )

    # sync the artifacts to minio
    file_count = 0
    for file in artifacts_dir.glob('**/*'):
        if file.is_file():
            key = str(file.relative_to(artifacts_dir))
            s3_minio.upload_file(str(file), key)
            file_count += 1
            if file_count % 10 == 0:
                print(f"  Syncing ... {file_count} files")

    print(f"  Synced {file_count} files to Minio")

def update_mlflow_config():
    """
    Update the MLflow config to use the local Minio
    """
    print("Updating MLflow config to use the local Minio ...")
    
    subprocess.run(['docker-compose', 'restart', 'mlflow-server'], check=True)

    print("MLflow config updated.")

def main():
    print("=" * 80)
    print("Restore S3 MLflow data to local")
    print("=" * 80)

    # --- 1. ensure the local MLflow server is running ---
    ensure_local_mlflow_running()

    # --- 2. download the Postgres dump from S3 ---
    dump_file = download_from_s3()

    # --- 3. restore the Postgres database from the dump file ---
    restore_to_local_postgres(dump_file)

    # --- 4. sync the artifacts to Minio ---
    response = input(f"Sync artifacts to Minio? (y/n): ")
    if response.lower() == 'y':
        sync_artifacts_to_minio()

    # --- 5. update the MLflow config to use the local Minio ---
    update_mlflow_config()

    print("=" * 80)
    print("\nRestore completed.")
    print("\nNext steps:")
    print("  Check the MLflow UI:")
    print("  http://localhost:5000")
    print("  Check the Minio UI:")
    print("  http://localhost:9000")
    print("  Clean up the local MLflow server:")
    print("  docker-compose down")

if __name__ == "__main__":
    main()