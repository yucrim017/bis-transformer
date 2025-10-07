import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from src.bistransformer.training.loop import train_one_experiment
from src.bistransformer.utils import metrics as _

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config")
        result = train_one_experiment(cfg, mlflow_run=run)
        for k, v in result["metrics"].items():
            mlflow.log_metric(k, float(v))
        
        n_params = sum(p.numel() for p in result["model"].parameters())
        mlflow.log_metric("model/num_params", n_params)

        if result.get("best_ckpt_path"):
            try:
                import mlflow.pytorch
                mlflow.pytorch.log_model(result["model"], artifact_path="model",
                                         registered_model_name="bis_transformer")
            except Exception:
                import mlflow.pytorch
                mlflow.pytorch.log_model(result["model"], artifact_path="model")

            mlflow.log_artifact(
                result["best_ckpt_path"],
                artifact_path="checkpoints",
            )


if __name__ == "__main__":
    main()