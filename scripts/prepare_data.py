from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import vitaldb as vdb
import hydra
from omegaconf import DictConfig, OmegaConf

from bistransformer.utils.data import (
    extract_1hz_features,
    build_tracks_from_cfg,
    columns_for_features,
    save_case_npz,
)

# ----------------------------
# case processing
# ----------------------------
def process_case(
    cid: int,
    tracks: List[str],
    output_dir: Path,
    cols: Dict[str, Any],
    fs: float,
    params: Dict[str, Any],
    save_npz: bool = True,
    save_parquet: bool = False,
    verbose: bool = False
    ) -> Dict[str, Any]:
    """
    Process a case and save the results to a file.

    Parameters
    ----------
    cid: int
        Case ID
    tracks: List[str]
        List of tracks to process
    output_dir: Path
        Output directory
    cols: Dict[str, Any]
        Dictionary of column names and their corresponding data
    fs: float
        Sampling frequency
    params: Dict[str, Any]
        Dictionary of parameters passed to the preprocessing function
    save_npz: bool
        Whether to save as npz file
    save_parquet: bool
        Whether to save as parquet file
    verbose: bool
        Whether to print verbose output

    Returns
    -------
    Dict[str, Any]
        Dictionary of results
    """
    if verbose:
        print(f"[prepare_data] cid={cid} ... fetching VitalDB")
        
    vf = vdb.VitalFile(cid, tracks)
    df_raw = vf.to_pandas(tracks, 1/fs)
    if df_raw.empty:
        raise RuntimeError(f"cid={cid}: empty dataframe")
    
    # --- extract 1Hz features ---
    kwargs = dict(params)
    kwargs["cols"] = cols
    df_feat = extract_1hz_features(df_raw, **kwargs)

    if verbose:
        print(f"  df_raw shape: {df_raw.shape}, df_feat shape: {df_feat.shape}")

    if df_feat.empty or ("BIS_target" not in df_feat.columns):
        raise RuntimeError(f"cid={cid}: no features or target generated")

    # --- process: df_feat -> X, y, T ---
    bis_col = cols["bis"]
    feat_cols = columns_for_features(df_feat, bis_col=bis_col)

    df_feat = df_feat.sort_values("sec").reset_index(drop=True)

    T = df_feat["sec"].to_numpy(dtype=np.int32)
    y = df_feat["BIS_target"].to_numpy(dtype=np.float32)[None, :]
    X = df_feat[feat_cols].to_numpy(dtype=np.float32).T

    # --- save ---
    out_dir = output_dir / f"case_{cid:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(
        cid=cid,
        seconds=int(y.shape[1]),
        n_features=X.shape[0],
        tracks=OmegaConf.to_container(tracks, resolve=True) \
            if OmegaConf.is_config(tracks) else list(tracks),
        cols_map=OmegaConf.to_container(cols, resolve=True) \
            if OmegaConf.is_config(cols) else cols,
        params=OmegaConf.to_container(params, resolve=True) \
            if OmegaConf.is_config(params) else params,
        features=list(feat_cols),
        bis_column=str(bis_col),
        )
    if save_npz:
        save_case_npz(out_dir / "features.npz", X=X, y=y, sec=T, META=meta)
    if save_parquet:
        df_feat.to_parquet(out_dir / "table.parquet", index=False)
    if verbose:
        print(f"[prepare_data] cid={cid} saved: secs={meta['seconds']}, feats={meta['n_features']}")

    return dict(
        cid=cid,
        seconds=meta["seconds"],
        n_features=meta["n_features"],
        save_dir=str(out_dir),
        has_npz=bool(save_npz),
        has_parquet=bool(save_parquet),
        )


# ----------------------------
# entrypoint
# ----------------------------
@hydra.main(version_base=None, config_path="../configs/prepare", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"[prepare_data] config:\n", OmegaConf.to_yaml(cfg))

    # setting from config
    tracks: List[str] = build_tracks_from_cfg(cfg.data)
    print(f"[prepare_data] tracks: {tracks}")
    output_dir: str = Path(cfg.prepare.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verbose: bool = bool(cfg.prepare.runtime.get("verbose", True))
    save_npz: bool = bool(cfg.prepare.runtime.get("save_npz", True))
    save_parquet: bool = bool(cfg.prepare.runtime.get("save_parquet", False))

    fs: float = cfg.prepare.params.get("fs", 128.0)

    # setting params passed to extract_1hz_features
    params = OmegaConf.to_container(cfg.prepare.params, resolve=True)

    cols: Dict[str, Any] = dict(
        bis=cfg.data.cols.get("bis", "BIS/BIS"),
        sqi=cfg.data.cols.get("sqi", "BIS/SQI"),
        emg=cfg.data.cols.get("emg", None),
        eeg=list(cfg.data.cols.get("eeg", [])),
        vital=list(cfg.data.cols.get("vital", [])),
        drug=list(cfg.data.cols.get("drug", [])),
    )
    
    # search case id from vitaldb dataset
    if cfg.prepare.id.get("cids"):
        cids = list(cfg.prepare.id.cids)
        if verbose:
            print(f"[prepare_data] use cids from config: {cids[:5]}{'...' if len(cids)>5 else ''}")
    else:
        cids = vdb.find_cases(tracks)
        if verbose:
            print(f"[prepare_data] found {len(cids)} cases")

    if len(cids) == 0:
        raise RuntimeError("[prepare_data] no cases found")

    if cfg.prepare.id.get("n_cases"):
        n_cases = min(cfg.prepare.id.n_cases, len(cids))
        if verbose:
            print(f"[prepare_data] use {n_cases} cases from config: [{', '.join(map(str, cids[:5]))}{', ...' if len(cids)>5 else ''}]")

    # process cases
    manifest_rows = []
    for cid in cids:
        if len(manifest_rows) >= n_cases or cid == cids[-1]:
            break
        try:
            row = process_case(
                cid=cid,
                tracks=tracks,
                output_dir=output_dir,
                cols=cols,
                fs=fs,
                params=params,
                save_npz=save_npz,
                save_parquet=save_parquet,
                verbose=verbose,
                )
            manifest_rows.append(row)
        except Exception as e:
            print(f"[prepare_data] skip cid={cid}: {e}")
            continue
        
    # save manifest
    man = pd.DataFrame(manifest_rows)
    man.to_csv(output_dir / "manifest.csv", index=False)
    if verbose:
        print(f"[prepare_data] saved manifest: {output_dir / 'manifest.csv'}")
        print(f"[prepare_data] done: {len(manifest_rows)} / {len(cids)} cases processed")

if __name__ == "__main__":
    main()