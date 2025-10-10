from __future__ import annotations
from typing import Tuple
import torch

from bistransformer.utils.metrics import mae, rmse, pearson

def _single_inference(
    model,
    batch,
    amp=True,
    device=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    single inference and return prediction and target
    """
    device = device
    x = batch["inputs"].to(device, non_blocking=True)
    y = batch["targets"].to(device, non_blocking=True)
    pred = model(x)
    return pred, y

def evaluate(
    model,
    loader,
    device,
    amp=True
) -> Tuple[float, float, float]:
    """
    evaluate model on loader, return metrics
    """
    device = device
    model.eval()
    m_mae = m_rmse = m_pear = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: \
                v.to(device, non_blocking=True) if hasattr(v, "to") \
                else v for k, v in batch.items()}
            pred, y = _single_inference(model, batch, amp, device)
            m_mae += mae(pred, y)
            m_rmse += rmse(pred, y)
            m_pear += pearson(pred, y)
    
    return m_mae / len(loader), m_rmse / len(loader), m_pear / len(loader)