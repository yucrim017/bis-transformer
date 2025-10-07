import os

import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from bistransformer.data.datamodule import build_dataloaders
from bistransformer.models.factory import build_model
from bistransformer.training.loop import _device, _eval_epoch


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    device = _device()
    _, _, test_loader = build_dataloaders(cfg)
    model = build_model(cfg.model).to(device)

    ckpt_path = getattr(
        cfg.train.ckpt, "path", None) or \
        os.path.join(
            getattr(cfg.train.ckpt, "dir", "outputs/checkpoints"),
            getattr(cfg.train.ckpt, "filename", "best.pt"),
        )
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)

    with mlflow.start_run() as run:
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config")

        metrics = _eval_epoch(
            model,
            test_loader,
            device,
            amp=bool(getattr(cfg.train, "amp", True))
        )
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
            
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

    print("[test]", {k: round(v, 6) for k, v in metrics.items()})

if __name__ == "__main__":
    main()