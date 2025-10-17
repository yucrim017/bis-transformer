import argparse
import yaml
from datetime import datetime
from pathlib import Path
from launch_training import launch_training_job


def load_config(config_path: str) -> dict:
    """Load config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker training job from config file")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Override job name"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Build image URI
    image_uri = (
        f"{config['aws']['account_id']}.dkr.ecr.{config['aws']['region']}.amazonaws.com/"
        f"{config['ecr']['repository_name']}:{config['ecr']['image_tag']}"
    )
    
    # ジョブ名を生成
    if args.job_name:
        job_name = args.job_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{config['training']['base_job_name']}-{timestamp}"
    
    # ハイパーパラメータを文字列に変換
    hyperparameters = {
        k: str(v) for k, v in config['hyperparameters'].items()
    }
    
    # タグを変換
    tags = [{"Key": k, "Value": str(v)} for k, v in config.get('tags', {}).items()]
    
    # MLflow設定
    mlflow_tracking_uri = config.get('mlflow', {}).get('tracking_uri')
    
    # トレーニングジョブを起動
    estimator = launch_training_job(
        job_name=job_name,
        image_uri=image_uri,
        role=config['aws']['role'],
        instance_type=config['training']['instance_type'],
        instance_count=config['training']['instance_count'],
        volume_size=config['training']['volume_size'],
        max_run=config['training']['max_run'],
        training_data_uri=config['data']['training_data'],
        output_path=config['data']['output_path'],
        hyperparameters=hyperparameters,
        base_job_name=config['training']['base_job_name'],
        mlflow_tracking_uri=mlflow_tracking_uri,
        tags=tags,
    )
    
    print("\n✅ Training job launched successfully!")
    print(f"Job name: {job_name}")


if __name__ == "__main__":
    main()

