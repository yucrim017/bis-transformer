from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from bistransformer.models.factory import build_model
from bistransformer.utils.data import load_npz_case, find_case_dir


def _ig_attributions(model, x):
    """
    Integrated Gradients Attribution
    https://captum.ai/api/attribution_methods/integrated_gradients.html

    return: (T, D), absolute attribution scores for each feature
    """
    try:
        from captum.attr import IntegratedGradients
        model.eval()

        def forward_fn(inp):
            # reduce to scalar per batch so IG works for both
            # scalar (B, 1) and sequence (B, T, 1) outputs
            out = model(inp)
            return out.reshape(out.shape[0], -1).mean(dim=1, keepdim=True)

        ig = IntegratedGradients(forward_fn)
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

def _plot_explain(feature_names, sm, ig, pi, fig_path):
    n_feat = len(feature_names)
    names = feature_names if n_feat else [f"feature_{i}" for i in range(sm.shape[1] if sm is not None else 0)]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    if sm is not None:
        sal = sm.mean(axis=0)
        order = np.argsort(-sal)
        axes[0].barh([names[i] for i in order][::-1], sal[order][::-1])
    axes[0].set_title("Saliency (mean |grad| over time)")

    if ig is not None:
        ig_mean = ig.mean(axis=0)
        order = np.argsort(-ig_mean)
        axes[1].barh([names[i] for i in order][::-1], ig_mean[order][::-1])
    axes[1].set_title("Integrated Gradients")

    if pi is not None:
        order = np.argsort(-pi)
        axes[2].barh([names[i] for i in order][::-1], pi[order][::-1])
    axes[2].set_title("Permutation Importance")

    if sm is not None:
        im = axes[3].imshow(sm.T, aspect="auto", cmap="viridis")
        axes[3].set_yticks(range(len(names)))
        axes[3].set_yticklabels(names, fontsize=7)
        axes[3].set_xlabel("time step in window")
        plt.colorbar(im, ax=axes[3], fraction=0.046)
    axes[3].set_title("Saliency heatmap (time x feature)")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(
        "cuda" if torch.cuda.is_available() \
        else "mps" if torch.backends.mps.is_available() \
        else "cpu"
    )
    ckpt_path = getattr(
        cfg.explain.model, "path", None) or \
        os.path.join(
            getattr(cfg.train.ckpt, "dir", "outputs/checkpoints"),
            getattr(cfg.train.ckpt, "best", "best.pt"),
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    d_in = ckpt["model"]["encoder.embed.0.weight"].shape[1]
    max_len = ckpt["model"]["encoder.pos.pe"].shape[1]

    case_dir = Path(cfg.explain.case_dir)
    if not case_dir.exists():
        case_dir = find_case_dir(Path("data/processed/v1"), d_in)
        print(f"[explain] case_dir not found, using: {case_dir}")

    data = load_npz_case(case_dir)
    feature_names = list(data["meta"].get("features", []))

    win = int(cfg.explain.window.length)
    Xc = data["X"].astype("float32")  # (D, T_total)
    mu = Xc.mean(axis=1, keepdims=True)
    std = Xc.std(axis=1, keepdims=True) + 1e-6
    Xn = (Xc - mu) / std
    start = Xn.shape[1] // 2
    x_win = Xn[:, start:start + win]  # (D, win)

    OmegaConf.set_struct(cfg.model, False)
    cfg.model.d_in = d_in
    cfg.model.max_len = max_len
    OmegaConf.set_struct(cfg.model, True)

    model = build_model(cfg.model).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    X = torch.from_numpy(x_win.T).unsqueeze(0).to(device)  # (1, win, D)

    out_dir = Path(cfg.explain.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- compute attributions ---
    ig = _ig_attributions(model, X)
    pi = _perm_importance(model, X) if model.head.target_mode == "scalar" else None
    sm = _saliency_map(model, X)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_id = getattr(cfg.explain, "run_id", None)

    with mlflow.start_run(run_id=run_id):
        if run_id is None:
            mlflow.set_tag("mode", "explain")
            mlflow.set_tag("case_dir", str(case_dir))
            mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config")

        if ig is not None:
            ig_path = out_dir / "ig_attributions.npy"
            np.save(ig_path, ig)
            print(f"[explain] IG saved: {ig_path}")
            mlflow.log_artifact(str(ig_path), artifact_path="explain")
            for i, score in enumerate(ig.mean(axis=0)):
                name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                mlflow.log_metric(f"IG/{name}", float(score))
        else:
            print("[explain] IG attribution failed")

        if pi is not None:
            pi_path = out_dir / "perm_importance.npy"
            np.save(pi_path, pi)
            print(f"[explain] Permutation importance saved: {pi_path}")
            mlflow.log_artifact(str(pi_path), artifact_path="explain")
            for i, score in enumerate(pi):
                name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                mlflow.log_metric(f"PermImportance/{name}", float(score))

        if sm is not None:
            sm_path = out_dir / "saliency_map.npy"
            np.save(sm_path, sm)
            print(f"[explain] Saliency map saved: {sm_path}")
            mlflow.log_artifact(str(sm_path), artifact_path="explain")
            for i, score in enumerate(sm.mean(axis=0)):
                name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                mlflow.log_metric(f"Saliency/{name}", float(score))

        # --- plot ---
        fig_path = out_dir / "explain.png"
        _plot_explain(feature_names, sm, ig, pi, fig_path)
        print(f"[explain] Figure saved: {fig_path}")
        mlflow.log_artifact(str(fig_path), artifact_path="explain")

        if run_id is None:
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")


if __name__ == "__main__":
    main()