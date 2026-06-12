from __future__ import annotations
from typing import Dict, Tuple, Any
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from bistransformer.utils.metrics import mae

def _variation_weighted_mse(
    pred: torch.Tensor,
    y: torch.Tensor,
    variation_weight: float = 0.0
    ) -> torch.Tensor:
    """
    MSE loss with extra weight on timesteps where the target changes a lot.

    For sequence targets (B, T, 1), weight = 1 + variation_weight * |dy/dt|.
    Falls back to plain MSE for scalar targets or variation_weight <= 0.
    """
    if variation_weight > 0 and y.dim() >= 2 and y.shape[1] > 1:
        dy = torch.zeros_like(y)
        dy[:, 1:] = (y[:, 1:] - y[:, :-1]).abs()
        weight = 1.0 + variation_weight * dy
        return (weight * (pred - y) ** 2).sum() / weight.sum()
    return F.mse_loss(pred, y)

def _forward_loss(
    model,
    batch,
    amp: bool=True,
    device: torch.device=None,
    variation_weight: float=0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    forward pass and loss calculation
    """
    x = batch["inputs"].to(device, non_blocking=True)
    y = batch["targets"].to(device, non_blocking=True)

    device_type = 'cuda' if device.type == 'cuda' else \
                  'mps' if device.type == 'mps' else 'cpu'

    if device_type == 'cuda' or device_type == 'cpu':
        with autocast(device_type, enabled=amp):
            pred = model(x)

            if pred.shape != y.shape:
                pred = pred.view(y.shape)
            loss = _variation_weighted_mse(pred, y, variation_weight)
        return pred, y, loss
    else: # mps
        with torch.autocast(device_type='mps', enabled=amp):
            pred = model(x)
            if pred.shape != y.shape:
                pred = pred.view(y.shape)
            loss = _variation_weighted_mse(pred, y, variation_weight)
        return pred, y, loss

def training_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    amp: bool=True,
    grad_clip: float=1.0,
    device: torch.device=None,
    variation_weight: float=0.0
    ) -> Tuple[float, float]:
    """
    receive hydra config and run training, return metrics and best model
    """
    model.train()
    total_loss = 0.0
    m_mae = 0.0

    for batch_idx, batch in enumerate(train_loader):

        optimizer.zero_grad(set_to_none=True)
        pred, y, loss = _forward_loss(model, batch, amp, device, variation_weight)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        m_mae += mae(pred, y)
        
    return total_loss / len(train_loader), m_mae / len(train_loader)