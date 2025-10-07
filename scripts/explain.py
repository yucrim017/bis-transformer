from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from bistransformer.models.factory import build_model
from bistransformer.training.loop import _device
from bistransformer.data.dataset import _load_npz_case


def _ig_attributions(model, x):
    """
    Integrated Gradients Attribution
    https://captum.ai/api/attribution_methods/integrated_gradients.html

    return: (T, D), absolute attribution scores for each feature
    """
    try:
        from captum.attr import IntegratedGradients
        model.eval()
        ig = IntegratedGradients(model)
        at = ig.attribute(x, target=None, n_steps=64)
        return at.squeeze(0).abs().cpu().numpy()
    except Exception as e:
        print(f"[explain] Error in IG attribution: {e}")
        return None

def _perm_importance(model, x):
    """
    Permutation Importance
    https://scikit-learn.org/stable/modules/permutation_importance.html

    return: (D,), absolute importance scores for each feature
    """
    T, D = x.shape[1], x.shape[2]
    with torch.no_grad():
        base = model(x).view(-1).item()
    out = np.zeros(D, dtype=np.float32)
    for d in range(D):
        xx = x.clone()
        idx = torch.randperm(T)
        xx[0, :, d] = x[0, idx, d]
        with torch.no_grad():
            val = model(xx).view(-1).item()
        out[d] = abs(val - base)
    return out

def _saliency_map(model, x):
    """
    Saliency Map
    https://captum.ai/api/attribution_methods/saliency.html

    return: (T, D), absolute saliency scores for each feature
    """
    try:
        model.eval()
        x.requires_grad_(True)
        out = model(x)
        out.backward(torch.ones_like(out))
        saliency = x.grad.abs()
        return saliency.squeeze(0).abs().cpu().numpy()
    except Exception:
        return None

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = _device()
    model = build_model(cfg.model).to(device)

    ckpt_path = getattr(
        cfg.train.ckpt, "path", None) or \
        os.path.join(
            getattr(cfg.train.ckpt, "dir", "outputs/checkpoints"),
            getattr(cfg.train.ckpt, "filename", "best.pt"),
        )
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    case_dir = Path(cfg.explain.case_dir)
    data = _load_npz_case(case_dir)
    X = torch.from_numpy(data["X"].astype("float32").T).unsqueeze(0).to(device)

    # --- integrated gradients ---
    ig = _ig_attributions(model, X)
    out_dir = Path(cfg.explain.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ig is not None:
        np.save(out_dir / "ig_attributions.npy", ig)
        print(f"[explain] IG saved: {out_dir/'ig_attributions.npy'}")
    else:
        print("[explain] IG attribution failed, falling back to permutation importance")
        # --- (fallback) permutation importance ---
        # this requires scalar head
        pi = _perm_importance(model, X) if model.target_mode == "scalar" else None
        if pi is not None:
            np.save(out_dir / "perm_importance.npy", pi)
            print(f"[explain] Permutation importance saved: {out_dir/'perm_importance.npy'}")

    # --- saliency map ---
    sm = _saliency_map(model, X)
    if sm is not None:
        np.save(out_dir / "saliency_map.npy", sm)
        print(f"[explain] Saliency map saved: {out_dir/'saliency_map.npy'}")


if __name__ == "__main__":
    main()