import logging

import os
from datetime import datetime
import time
import random
import json

import numpy as np
import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from bistransformer.dataset import build_dataloaders
from bistransformer.models import build_model
from bistransformer.training import (
    training_epoch,
    build_optimizer,
    build_scheduler,
    EarlyStopping,
    CheckpointSaver,
)
from bistransformer.eval import evaluate

log = logging.getLogger(__name__)

def train_one_experiment(cfg: DictConfig):
    """
    training model and evaluate on single MLflow run
    """
    log.info(f"Training model with config: {cfg.train}")
    log.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("="*80)

    train_loader, val_loader, _ = build_dataloaders(cfg)

    first_batch = next(iter(train_loader))
    # automatically set model input size and max length
    d_in = first_batch["inputs"].shape[-1]
    max_len = first_batch["inputs"].shape[1]

    OmegaConf.set_struct(cfg.model, False)

    cfg.model.d_in = d_in
    cfg.model.max_len = max_len

    OmegaConf.set_struct(cfg.model, True)

    # build model
    model = build_model(cfg.model)

    log.info("Model Information:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total parameters: {total_params:,}")
    log.info(f"  Trainable parameters: {trainable_params:,}")
    log.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    log.info("="*80)

    grad_clip_norm = cfg.train.grad_clip_norm
    amp = cfg.train.amp
    variation_weight = float(getattr(cfg.train.loss, "variation_weight", 0.0)) \
        if hasattr(cfg.train, "loss") else 0.0

    # build optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    early = EarlyStopping(
        monitor="Val/MAE",
        patience=int(getattr(cfg.train, "patience", 8)),
    )

    best_ckpt_saver = CheckpointSaver(
        dirpath=str(getattr(cfg.train.ckpt, "dir", "outputs/checkpoints")),
        filename=str(getattr(cfg.train.ckpt, "best", "best.pt")),
    )
    last_ckpt_saver = CheckpointSaver(
        dirpath=str(getattr(cfg.train.ckpt, "dir", "outputs/checkpoints")),
        filename=str(getattr(cfg.train.ckpt, "last", "last.pt")),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() \
        else "mps" if torch.backends.mps.is_available() \
        else "cpu"
    )

    model.to(device)

    epochs = cfg.train.epochs
    eval_interval = cfg.train.eval_interval
    best_ckpt_path = None
    last_ckpt_path = None
    val_metrics = {"MAE": float("inf"), "RMSE": float("inf"), "Pearson": 0.0}

    start_epoch = 1
    last_ckpt_path_existing = last_ckpt_saver.ckpt_path
    if os.path.exists(last_ckpt_path_existing):
        log.info(f"Resuming from checkpoint: {last_ckpt_path_existing}")
        ckpt = torch.load(last_ckpt_path_existing, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        early.best = ckpt["early"]["best"]
        early.count = ckpt["early"]["count"]

        rng = ckpt.get("rng", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s for s in rng["cuda"]])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])

        start_epoch = ckpt["epoch"] + 1
        log.info(f"Resumed at epoch {start_epoch} (early.best={early.best:.6f}, early.count={early.count})")

    start_run_time = time.time()

    log.info("Start Training ...")

    for epoch in range(start_epoch, epochs + 1):
        start_epoch_time = time.time()

        model.train()
        loss, mae = training_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            amp,
            grad_clip_norm,
            device,
            variation_weight
        )
        epoch_time = time.time() - start_epoch_time
        
        log.info(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {loss:.4f} | "
            f"Train/MAE: {mae:.4f} | "
            f"Time: {epoch_time:.1f}s ({epoch_time*1000:.1f}ms)"
        )
        mlflow.log_metric("Train/Loss", loss, step=epoch)
        mlflow.log_metric("Train/MAE", mae, step=epoch)
        mlflow.log_metric("Time/Epoch", epoch_time, step=epoch)
        mlflow.log_metric("Time/Avg_step", epoch_time / len(train_loader), step=epoch)
        mlflow.log_metric("Time/Elapsed", time.time() - start_run_time, step=epoch)

        if epoch % eval_interval == 0:

            model.eval()
            with torch.no_grad():
                val_mae, val_rmse, val_pearson = evaluate(
                    model,
                    val_loader,
                    device,
                    amp=bool(getattr(cfg.train, "amp", True))
                )
                val_metrics = {
                    "MAE": val_mae,
                    "RMSE": val_rmse,
                    "Pearson": val_pearson,
                }
            log.info(
                f"Val metrics:\n  "
                f"MAE: {val_metrics['MAE']:.6f} | "
                f"RMSE: {val_metrics['RMSE']:.6f} | "
                f"Pearson: {val_metrics['Pearson']:.6f}"
            )
            for k, v in val_metrics.items():
                mlflow.log_metric(f"Val/{k}", float(v), step=epoch)
        
        if hasattr(scheduler, "step"):
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_metrics["MAE"])
            else:
                scheduler.step()

        # save last checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "early": {
                "best": early.best,
                "count": early.count,
            },
            "rng": {
                "torch": torch.get_rng_state().cpu(),
                "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            }
        }

        last_ckpt_path = last_ckpt_saver.save(ckpt)
        
        if val_metrics["MAE"] <= early.best:
            best_ckpt_path = best_ckpt_saver.save(ckpt)

        if early.step(val_metrics["MAE"]):
            log.info(f"Early stopping at epoch {epoch}")
            break
    
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model"])

    return {
        "mae_best": early.best,
        "best_ckpt_path": best_ckpt_path,
        "last_ckpt_path": last_ckpt_path,
        "model": model,
    }

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp is None:
        exp_id = client.create_experiment(cfg.mlflow.experiment_name)
    else:
        exp_id = exp.experiment_id
    run = client.create_run(exp_id)
    print(run.info.run_id)

    with mlflow.start_run(run_id=run.info.run_id):
        # Log full config as artifact
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config")
        
        # Log key parameters for easy filtering/comparison in MLflow UI
        mlflow.log_params({
            # Model params
            "model.name": cfg.model.name,
            "model.d_in": cfg.model.d_in,
            "model.max_len": cfg.model.max_len,
            "model.target_mode": cfg.model.target_mode,
            "model.encoder.d_model": cfg.model.encoder.d_model,
            "model.encoder.attention_layers": cfg.model.encoder.attention_layers,
            "model.encoder.n_head": cfg.model.encoder.n_head,
            "model.encoder.d_ff": cfg.model.encoder.d_ff,
            "model.encoder.dropout": cfg.model.encoder.dropout,
            "model.head.hidden_size": cfg.model.head.hidden_size,
            "model.head.pool": cfg.model.head.pool,
            
            # Training params
            "train.epochs": cfg.train.epochs,
            "train.optimizer": cfg.train.optimizer.name,
            "train.lr": cfg.train.optimizer.lr,
            "train.weight_decay": cfg.train.optimizer.weight_decay,
            "train.scheduler": cfg.train.scheduler.name,
            "train.batch_size": cfg.data.loader.batch_size,
            "train.grad_clip": cfg.train.grad_clip_norm,
            "train.patience": cfg.train.patience,
            "train.amp": cfg.train.amp,
            "train.loss.variation_weight": float(getattr(cfg.train.loss, "variation_weight", 0.0)) \
                if hasattr(cfg.train, "loss") else 0.0,
            
            # Data params
            "data.window.length": cfg.data.window.length,
            "data.window.hop": cfg.data.window.hop,
            "data.normalization": cfg.data.normalization.enable,
        })

        result = train_one_experiment(cfg)

        mlflow.log_metric("mae_best", result["mae_best"])

        n_params = sum(p.numel() for p in result["model"].parameters())
        mlflow.log_metric("num_params", n_params)

        if result.get("best_ckpt_path"):
            # save checkpoint file as light weight for inference
            mlflow.log_artifact(result["best_ckpt_path"], "checkpoints")

            # save best checkpoint as PyTorch model
            try:
                mlflow.pytorch.log_model(
                    pytorch_model=result["model"],
                    artifact_path="model",
                )
            except Exception as e:
                log.warning(f"Failed to log PyTorch model to MLflow: {e}")

            # remove best checkpoint file after logged
            os.remove(result["best_ckpt_path"])


if __name__ == "__main__":
    main()