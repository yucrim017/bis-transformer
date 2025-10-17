"""
SageMaker Training Job Launcher
ローカル環境からSageMakerでbis-transformerの学習ジョブを起動
"""

import argparse
import os
import json
from datetime import datetime
import boto3
import sagemaker
from sagemaker.estimator import Estimator


def get_execution_role(role_name=None):
    """IAMロールを取得"""
    if role_name:
        return role_name
    
    # SageMakerセッションからロールを取得
    try:
        role = sagemaker.get_execution_role()
        return role
    except:
        # ローカル環境の場合、環境変数から取得
        role = os.environ.get('SAGEMAKER_ROLE')
        if not role:
            raise ValueError(
                "SageMaker execution role not found. "
                "Please specify --role or set SAGEMAKER_ROLE environment variable."
            )
        return role


def launch_training_job(
    job_name: str,
    image_uri: str,
    role: str,
    instance_type: str,
    instance_count: int,
    volume_size: int,
    max_run: int,
    training_data_uri: str,
    output_path: str,
    hyperparameters: dict,
    base_job_name: str = None,
    mlflow_tracking_uri: str = None,
    tags: list = None,
):
    """
    SageMakerトレーニングジョブを起動
    
    Parameters
    ----------
    job_name : str
        ジョブ名
    image_uri : str
        DockerイメージのECR URI
    role : str
        SageMaker実行ロールARN
    instance_type : str
        インスタンスタイプ（例: ml.p3.2xlarge）
    instance_count : int
        インスタンス数
    volume_size : int
        EBSボリュームサイズ (GB)
    max_run : int
        最大実行時間（秒）
    training_data_uri : str
        学習データのS3 URI
    output_path : str
        出力先のS3パス
    hyperparameters : dict
        ハイパーパラメータ
    base_job_name : str, optional
        ベースジョブ名（ジョブ名の接頭辞）
    mlflow_tracking_uri : str, optional
        MLflow Tracking Server URI
    tags : list, optional
        リソースタグ
    """
    
    # SageMakerセッションを作成
    sess = sagemaker.Session()
    
    # Estimatorを作成
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size=volume_size,
        max_run=max_run,
        output_path=output_path,
        base_job_name=base_job_name or "bis-transformer",
        sagemaker_session=sess,
        hyperparameters=hyperparameters,
        tags=tags or [],
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {"Name": "train:loss", "Regex": "Train/Loss: ([0-9\\.]+)"},
            {"Name": "train:mae", "Regex": "Train/MAE: ([0-9\\.]+)"},
            {"Name": "val:mae", "Regex": "MAE: ([0-9\\.]+)"},
            {"Name": "val:rmse", "Regex": "RMSE: ([0-9\\.]+)"},
            {"Name": "val:pearson", "Regex": "Pearson: ([0-9\\.]+)"},
        ],
    )
    
    # MLflow設定を環境変数として渡す
    if mlflow_tracking_uri:
        estimator.set_hyperparameters(
            **estimator.hyperparameters(),
            **{"mlflow-tracking-uri": mlflow_tracking_uri}
        )
    
    print("=" * 80)
    print("SageMaker Training Job Configuration:")
    print(f"  Job Name: {job_name}")
    print(f"  Image URI: {image_uri}")
    print(f"  Instance Type: {instance_type}")
    print(f"  Instance Count: {instance_count}")
    print(f"  Volume Size: {volume_size} GB")
    print(f"  Max Run: {max_run} seconds")
    print(f"  Training Data: {training_data_uri}")
    print(f"  Output Path: {output_path}")
    print(f"  Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
    print("=" * 80)
    
    # トレーニングジョブを起動
    estimator.fit(
        inputs={"training": training_data_uri},
        job_name=job_name,
        wait=False,  # 非同期で起動
    )
    
    print(f"\nTraining job '{job_name}' has been launched!")
    print(f"Monitor progress at: https://console.aws.amazon.com/sagemaker/home?region={sess.boto_region_name}#/jobs/{job_name}")
    
    return estimator


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker training job for bis-transformer")
    
    # 必須パラメータ
    parser.add_argument(
        "--image-uri",
        type=str,
        required=True,
        help="ECR image URI (e.g., 123456789012.dkr.ecr.us-east-1.amazonaws.com/bis-transformer:latest)"
    )
    parser.add_argument(
        "--training-data",
        type=str,
        required=True,
        help="S3 URI for training data (e.g., s3://my-bucket/data/processed/)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="S3 path for output (e.g., s3://my-bucket/output/)"
    )
    
    # オプションパラメータ
    parser.add_argument("--role", type=str, default=None, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", type=str, default="ml.p3.2xlarge", help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--volume-size", type=int, default=50, help="EBS volume size (GB)")
    parser.add_argument("--max-run", type=int, default=86400, help="Max run time in seconds (default: 24 hours)")
    parser.add_argument("--job-name", type=str, default=None, help="Training job name")
    parser.add_argument("--base-job-name", type=str, default="bis-transformer", help="Base job name")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    
    # ハイパーパラメータ
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--attention-layers", type=int, default=4, help="Number of attention layers")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    
    # タグ
    parser.add_argument("--tags", type=str, default=None, help="Tags in JSON format")
    
    args = parser.parse_args()
    
    # ジョブ名を生成（指定がない場合）
    if args.job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.job_name = f"{args.base_job_name}-{timestamp}"
    
    # 実行ロールを取得
    role = get_execution_role(args.role)
    
    # ハイパーパラメータを構築
    hyperparameters = {
        "train.epochs": str(args.epochs),
        "data.loader.batch_size": str(args.batch_size),
        "train.optimizer.lr": str(args.lr),
        "train.optimizer.weight_decay": str(args.weight_decay),
        "model.encoder.d_model": str(args.d_model),
        "model.encoder.attention_layers": str(args.attention_layers),
        "model.encoder.n_head": str(args.n_head),
    }
    
    # タグをパース
    tags = []
    if args.tags:
        tags_dict = json.loads(args.tags)
        tags = [{"Key": k, "Value": v} for k, v in tags_dict.items()]
    
    # トレーニングジョブを起動
    estimator = launch_training_job(
        job_name=args.job_name,
        image_uri=args.image_uri,
        role=role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size=args.volume_size,
        max_run=args.max_run,
        training_data_uri=args.training_data,
        output_path=args.output_path,
        hyperparameters=hyperparameters,
        base_job_name=args.base_job_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        tags=tags,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

