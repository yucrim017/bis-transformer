from __future__ import annotations
from typing import Dict, Any
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from bistransformer.models.factory import build_model
from bistransformer.data.datamodule import build_dataloaders
from bistransformer.utils.metrics import mae, rmse, pearson
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from .callbacks import EarlyStopping, CheckpointSaver


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and \
        torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _forward_loss(model, batch, amp: bool=True, device: torch.device=None):
    """
    forward pass and loss calculation
    """
    x = batch["X"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)

    device_type = 'cuda' if device.type == 'cuda' else \
                  'mps' if device.type == 'mps' else 'cpu'

    if device_type == 'cuda' or device_type == 'cpu':
        with autocast(device_type, enabled=amp):
            pred = model(x)
            
            if pred.shape != y.shape:
                pred = pred.view(y.shape)
            loss = F.mse_loss(pred, y)
        return pred, y, loss
    else: # mps
        with torch.autocast(device_type='mps', enabled=amp):
            pred = model(x)
            if pred.shape != y.shape:
                pred = pred.view(y.shape)
            loss = F.mse_loss(pred, y)
        return pred, y, loss

def _eval_epoch(
    model, 
    loader, 
    device: torch.device=None, 
    amp: bool=True
) -> Dict[str, float]:
    """
    evaluate model on loader, return metrics
    """
    device = device or _device()
    model.eval()
    m_mae = m_rmse = m_pear = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: \
                v.to(device, non_blocking=True) if hasattr(v, "to") \
                else v for k, v in batch.items()}
            pred, y, _ = _forward_loss(model, batch, amp, device)
            m_mae += mae(pred, y)
            m_rmse += rmse(pred, y)
            m_pear += pearson(pred, y)
            n += 1
    return {
        "mae": m_mae / max(n, 1),
        "rmse": m_rmse / max(n, 1),
        "pearson": m_pear / max(n, 1),
    }

def train_one_experiment(cfg, mlflow_run=None) -> Dict[str, Any]:
    """
    receive hydra config and run training, return metrics and best model
    """
    device = _device()
    amp = bool(getattr(cfg.train, "amp", True))
    grad_clip = float(getattr(cfg.train, "grad_clip_norm", 1.0))
    epochs = int(getattr(cfg.train, "epochs", 50))

    train_loader, val_loader, test_loader = \
        build_dataloaders(cfg)

    first_batch = next(iter(train_loader))
    d_in = first_batch["inputs"].shape[-1]
    cfg.model.d_in = d_in

    model = build_model(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    early = EarlyStopping(
        monitor="val_mae",
        patience=int(getattr(cfg.train, "patience", 8)),
    )
    ckpt_saver = CheckpointSaver(
        dirpath=str(getattr(cfg.train.ckpt, "dir", "outputs/checkpoints")),
        filename=str(getattr(cfg.train.ckpt, "filename", "best.pt")),
    )
    scaler = GradScaler(enabled=amp)

    best_path = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: \
                v.to(device, non_blocking=True) if hasattr(v, "to") \
                else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred, y, loss = _forward_loss(model, batch, amp, device)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            val_metrics = _eval_epoch(model, val_loader, device, amp)
            if hasattr(scheduler, "step") and \
                scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_metrics["mae"])
            else:
                scheduler.step()
            if mlflow_run is not None:
                import mlflow
                mlflow.log_metric(
                    "train/loss", 
                    total_loss / max(1, len(train_loader)),
                    step=epoch,
                )
                mlflow.log_metric(
                    "val/mae",
                    val_metrics["mae"],
                    step=epoch,
                )
                mlflow.log_metric(
                    "val/rmse",
                    val_metrics["rmse"],
                    step=epoch,
                )
                mlflow.log_metric(
                    "val/pearson",
                    val_metrics["pearson"],
                    step=epoch,
                )
            
            if val_metrics["mae"] <= early.best:
                best_path = ckpt_saver.save(model.state_dict())

            if early.step(val_metrics["mae"]):
                break

        if best_path and os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        
        test_metrics = _eval_epoch(model, test_loader, device, amp)
        if mlflow_run is not None:
            mlflow.log_metric(
                "test/mae",
                test_metrics["mae"],
                step=epoch,
            )
            mlflow.log_metric(
                "test/rmse",
                test_metrics["rmse"],
                step=epoch,
            )
            mlflow.log_metric(
                "test/pearson",
                test_metrics["pearson"],
                step=epoch,
            )
    
    result = {
        "metrics": {
            "val/mae_best": early.best,
            "test/mae": test_metrics["mae"],
            "test/rmse": test_metrics["rmse"],
            "test/pearson": test_metrics["pearson"],
        },
        "best_ckpt_path": best_path or "",
        "model": model,
        }
    return result