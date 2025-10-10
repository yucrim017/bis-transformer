from __future__ import annotations

from pathlib import Path
import os
import json

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from bistransformer.dataset import WindowConfig
from bistransformer.models.factory import build_model
from bistransformer.eval import evaluate
from bistransformer.utils.data import load_npz_case

def sliding_window(T, win, hop):
    return list(range(0, max(0, T - win + 1), hop))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(
        "cuda" if torch.cuda.is_available() \
        else "mps" if torch.backends.mps.is_available() \
        else "cpu"
    )
    model = build_model(cfg.model).to(device)

    ckpt_path = getattr(
        cfg.predict.model, "path", None) or \
        os.path.join(
            getattr(cfg.train.ckpt, "dir", "outputs/checkpoints"),
            getattr(cfg.train.ckpt, "last", "last.pt"),
        )

    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    data = load_npz_case(Path(cfg.predict.case_dir))
    X, sec = data["X"].astype("float32"), data["sec"].astype("float32")
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
    np.savetxt(
        out_dir / "prediction.csv",
        np.vstack([sec, out]).T,
        delimiter=",",
        fmt=["%d","%.6f"],
        header="sec,pred_bis",
        comments="",
    )
    print(f"[predict] saved: {out_dir/'prediction.csv'}")


if __name__ == "__main__":
    main()