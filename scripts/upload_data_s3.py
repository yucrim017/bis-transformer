import argparse
from pathlib import Path
import boto3

def upload_data_s3(
    bucket_name: str,
    region_name: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    data_dir: str
) -> None:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    directory = Path(data_dir)
    for file in directory.glob("**/*"):
        if file.is_file():
            key = str(file.relative_to(directory))
            s3.upload_file(
                str(file),
                bucket_name,
                f"{data_dir}/{key}"
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-name", type=str, default="bis-transformer")
    parser.add_argument("--region-name", type=str, default="us-east-1")
    parser.add_argument("--aws-access-key-id", type=str, default="minio")
    parser.add_argument("--aws-secret-access-key", type=str, default="dev-password")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    args = parser.parse_args()
    print(f"Uploading data from {args.data_dir} to {args.bucket_name}")
    upload_data_s3(args.bucket_name, args.region_name, args.aws_access_key_id, args.aws_secret_access_key, args.data_dir)

if __name__ == "__main__":
    main()