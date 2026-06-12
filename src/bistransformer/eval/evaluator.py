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
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: \
                v.to(device, non_blocking=True) if hasattr(v, "to") \
                else v for k, v in batch.items()}
            pred, y = _single_inference(model, batch, amp, device)
            preds.append(pred)
            targets.append(y)

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return mae(preds, targets), rmse(preds, targets), pearson(preds, targets)