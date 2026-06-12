import os

import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from bistransformer.dataset import build_dataloaders
from bistransformer.models import build_model
from bistransformer.eval import evaluate

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    device = torch.device(
        "cuda" if torch.cuda.is_available() \
        else "mps" if torch.backends.mps.is_available() \
        else "cpu"
    )

    _, _, test_loader = build_dataloaders(cfg)

    first_batch = next(iter(test_loader))
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.d_in = first_batch["inputs"].shape[-1]
    cfg.model.max_len = first_batch["inputs"].shape[1]
    OmegaConf.set_struct(cfg.model, True)

    model = build_model(cfg.model).to(device)

    ckpt_path = os.path.join(
        getattr(cfg.train.ckpt, "dir", "outputs/checkpoints"),
        getattr(cfg.train.ckpt, "best", "best.pt"),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)

    run_id = getattr(cfg.evaluate, "run_id", None)

    with mlflow.start_run(run_id=run_id) as run:
        if run_id is None:
            # standalone evaluation run: tag mode and keep config for reference
            mlflow.set_tag("mode", "evaluate")
            mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config")

        test_mae, test_rmse, test_pearson = evaluate(
            model,
            test_loader,
            device,
            amp=bool(getattr(cfg.train, "amp", True))
        )
        metrics = {"MAE": test_mae, "RMSE": test_rmse, "Pearson": test_pearson}
        for k, v in metrics.items():
            mlflow.log_metric(f"Test/{k}", float(v))

        if run_id is None:
            # no source training run to reference: keep a copy of the checkpoint
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

    print("[test]", {k: round(v, 6) for k, v in metrics.items()})

if __name__ == "__main__":
    main()