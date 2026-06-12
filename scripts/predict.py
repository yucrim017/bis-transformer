from __future__ import annotations

from pathlib import Path
import os
import json

import numpy as np
import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from bistransformer.dataset import WindowConfig
from bistransformer.models.factory import build_model
from bistransformer.utils.data import load_npz_case, find_case_dir
from bistransformer.utils.metrics import mae, rmse, pearson

def sliding_window(T, win, hop):
    return list(range(0, max(0, T - win + 1), hop))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(
        "cuda" if torch.cuda.is_available() \
        else "mps" if torch.backends.mps.is_available() \
        else "cpu"
    )
    ckpt_path = getattr(
        cfg.predict.model, "path", None) or \
        os.path.join(
            getattr(cfg.train.ckpt, "dir", "outputs/checkpoints"),
            getattr(cfg.train.ckpt, "last", "last.pt"),
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    d_in = ckpt["model"]["encoder.embed.0.weight"].shape[1]
    max_len = ckpt["model"]["encoder.pos.pe"].shape[1]

    OmegaConf.set_struct(cfg.model, False)
    cfg.model.d_in = d_in
    cfg.model.max_len = max_len
    OmegaConf.set_struct(cfg.model, True)

    model = build_model(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    case_dir = Path(cfg.predict.case_dir)
    if not case_dir.exists():
        case_dir = find_case_dir(Path("data/processed/v1"), d_in)
        print(f"[predict] case_dir not found, using: {case_dir}")

    data = load_npz_case(case_dir)
    X, sec = data["X"].astype("float32"), data["sec"].astype("float32")

    if bool(getattr(cfg.data.normalization, "enable", False)):
        mu = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + float(getattr(cfg.data.normalization, "eps", 1e-6))
        X = (X - mu) / std

    T = X.shape[1]
    win = int(cfg.predict.window.length)
    hop = int(cfg.predict.window.hop)

    preds = np.full(T, np.nan, dtype=np.float32)
    counts = np.zeros(T, dtype=np.int32)

    with torch.no_grad():
        for st in sliding_window(T, win, hop):
            x = torch.from_numpy(X[:, st:st+win].T).unsqueeze(0).to(device)
            y = model(x).view(-1)
            if y.numel() == 1: # scalar head
                preds[st+win-1] = y.item()
                counts[st+win-1] += 1
            else: # sequence head
                y = y.cpu().numpy()
                preds[st:st+win] = \
                    np.nanmean(np.vstack([preds[st:st+win], y]), axis=0) \
                    if not np.isnan(preds[st:st+win]).all() else y
                counts[st:st+win] += 1
    
    mask = counts > 0
    out = np.interp(np.arange(T), np.where(mask)[0], preds[mask]) \
        if mask.any() else preds

    out_dir = Path(cfg.predict.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv_path = out_dir / "prediction.csv"
    np.savetxt(
        pred_csv_path,
        np.vstack([sec, out]).T,
        delimiter=",",
        fmt=["%d","%.6f"],
        header="sec,pred_bis",
        comments="",
    )
    print(f"[predict] saved: {pred_csv_path}")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_id = getattr(cfg.predict, "run_id", None)

    with mlflow.start_run(run_id=run_id):
        if run_id is None:
            mlflow.set_tag("mode", "predict")
            mlflow.set_tag("case_dir", str(cfg.predict.case_dir))
            mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config")

        y_true = data["y"].astype("float32").reshape(-1)
        valid = mask & ~np.isnan(y_true)
        if valid.any():
            pred_t = torch.from_numpy(out[valid])
            true_t = torch.from_numpy(y_true[valid])
            mlflow.log_metric("Predict/MAE", mae(pred_t, true_t))
            mlflow.log_metric("Predict/RMSE", rmse(pred_t, true_t))
            mlflow.log_metric("Predict/Pearson", pearson(pred_t, true_t))

        mlflow.log_artifact(str(pred_csv_path), artifact_path="predict")
        if run_id is None:
            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")


if __name__ == "__main__":
    main()