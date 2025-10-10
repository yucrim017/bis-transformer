from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Sequence
import json

import numpy as np
import pandas as pd
from scipy.signal import welch

from bistransformer.utils.kwargs import apply_kwargs

_DEFAULTS: Dict[str, Any] = dict(
    fs=128,
    cols=None,

    # defaults for reindex_bis_and_trim_head
    treat_zero_as_nan=True,
    add_sec_col=True,
    reset_row_index=False,

    # defaults for eeg_bandpower_1hz
    window_sec=4,
    bands=[
        [0.5, 4, "delta"],
        [4, 8, "theta"],
        [8, 13, "alpha"],
        [13, 30, "beta"],
        [30, 47, "gamma"]
    ],
    main_band=[0.5, 47.0],
    use_relative=True,
    use_log=True,
    min_coverage=0.8,
    eps=1e-12,

    # defaults for extract_1hz_features
    bis_ffill_seconds=1,
    sqi_threshold=0.5,
    drop_zero_bis=True,

    lag_seconds=15 # >0 for online processing
)


def columns_for_features(df_feat: pd.DataFrame, bis_col: str) -> List[str]:
    """extract columns for features"""
    exclude = {"sec", "BIS_target", bis_col}
    return [c for c in df_feat.columns if c not in exclude]

def save_case_npz(path: Path, **arrays: np.ndarray) -> None:
    """save arrays to npz file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        if isinstance(v, (dict, list, tuple)):
            arrays[k] = \
                np.array([json.dumps(v, ensure_ascii=False)], dtype=object)
    np.savez_compressed(path, **arrays)

def load_npz_case(case_dir: Path) -> Dict[str, Any]:
    """load case from npz file"""
    f = np.load(case_dir / "features.npz", allow_pickle=True)
    X = f["X"]
    y = f["y"]
    sec = f["sec"]
    meta = {}
    if "META" in f.files:
        try:
            meta_data = f["META"]
            if isinstance(meta_data, np.ndarray) and meta_data.dtype == object:
                # JSON文字列として保存されている場合
                import json
                meta = json.loads(meta_data.item())
            elif isinstance(meta_data, np.ndarray):
                meta = meta_data.item()
            else:
                meta = meta_data
        except Exception:
            meta = {}
    return dict(
        X=X,
        y=y,
        sec=sec,
        meta=meta,
    )


def reindex_bis_and_trim_head(
    df: pd.DataFrame,
    **kwargs
    ) -> pd.DataFrame:
    """
    reindex rows based on first valid BIS and trim head

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe containing BIS data
    fs: int
        Sampling frequency
    cols: Dict[str, Any]
        Dictionary of column names and their corresponding data
        columns:
            - "bis": str
                Column name for BIS data
    """
    cfg = apply_kwargs(_DEFAULTS, **kwargs)
    if cfg["cols"] is None or "bis" not in cfg["cols"]:
        raise ValueError("cols={'bis':[...]} is required")

    bis = str(cfg["cols"]["bis"])
    fs = int(cfg["fs"])

    if bis not in df.columns:
        raise ValueError(f"BIS column {bis} not found in dataframe")
    
    s = df[bis]
    valid_bis = s.notna()
    if cfg["treat_zero_as_nan"]:
        valid_bis &= (s != 0)
    
    if not valid_bis.any():
        raise ValueError("No valid BIS data found")

    # reindex based on first valid BIS
    first_idx = valid_bis.idxmax()
    # trim head
    df2 = df.loc[first_idx:].copy()

    # --- add sec column ---
    if cfg["add_sec_col"]:
        if isinstance(df2.index, pd.DatetimeIndex):
            df2["sec"] = \
                ((df2.index - df2.index[0]).total_seconds()).astype("int32")
        else:
            rel = (df2.index.values - df2.index[0]).astype("float32")
            df2['sec'] = np.floor(rel / float(fs))

    # --- reset index ---
    if cfg["reset_row_index"]:
        df2.reset_index(drop=True, inplace=True)

    return df2

def eeg_bandpower_1hz(
    eeg_seq: pd.DataFrame,
    sec_seq: Sequence[int],
    **kwargs
    ) -> pd.DataFrame:
    """
    calculate EEG bandpower at 1 Hz resolution

    Parameters
    ----------
    eeg_seq: pd.DataFrame
        dataframe containing EEG sequence
    sec_seq: Sequence[int]
        Sequence seconds including unique values
    fs: int
        Sampling frequency
    cols: Dict[str, Any]
        Dictionary of column names and their corresponding data
        columns:
            - "eeg": Sequence[str]
                Column names for EEG data
    window_sec: int
        Window size in seconds
    bands: tuple[tuple[float, float, str]]
        List of bands
    main_band: tuple[float, float]
        Main band
    use_relative: bool
        Whether to use relative bandpower
    use_log: bool
        Whether to use log bandpower
    min_coverage: float
        Minimum coverage
    eps: float
        Epsilon
    """
    cfg = apply_kwargs(_DEFAULTS, **kwargs)

    if cfg["cols"] is None or "eeg" not in cfg["cols"]:
        return pd.DataFrame(columns=["sec"])

    fs = int(cfg["fs"])
    eeg = list(cfg["cols"]["eeg"])
    if eeg is None or len(eeg) == 0:
        return pd.DataFrame(columns=["sec"])

    window_sec = int(cfg["window_sec"])
    
    # cast to float all band values
    bands_raw = tuple(cfg["bands"])
    bands = tuple((float(lo), float(hi), str(name)) for lo, hi, name in bands_raw)
    main_lo, main_hi = cfg["main_band"]
    main_band = (float(main_lo), float(main_hi))

    min_coverage = float(cfg["min_coverage"])
    eps = float(cfg["eps"])

    def base(col: str) -> str:
        return col[:-4] + "_" if col.endswith("WAV") else (col + "_")

    n = len(eeg_seq)
    if n == 0:
        return pd.DataFrame(columns=["sec"])

    win = int(round(window_sec * fs))
    need = max(1, int(round(win * min_coverage)))
    # Filter out NA values and duplicate values, cast to int from sec_seq
    sec_arr = \
        pd.to_numeric(pd.Series(sec_seq), errors="coerce")\
        .dropna()\
        .unique()\
        .astype(int)
    sec_arr.sort()
    eeg_seq = eeg_seq.reset_index(drop=True)

    out: List[Dict[str, Any]] = []
    for sec in sec_arr:
        i = int(round(sec * fs))
        i0 = max(0, i - win)
        i1 = min(i, n)
        row = {"sec": sec}

        # --- calculate bandpower ---
        if i1 - i0 >= need:
            for c in eeg:
                x = eeg_seq.iloc[i0:i1][c]
                x = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
                if len(x) < need: continue # skip if not enough data
                f, P = welch(x, fs=fs, nperseg=min(len(x), int(fs)), detrend="constant")
                m_main = (f >= main_band[0]) & (f < main_band[1])
                p_total = P[m_main].sum() if m_main.any() else np.nan
                b = base(c)

                for lo, hi, name in bands:
                    m = (f >= lo) & (f < hi)
                    p = P[m].sum() if m.any() else np.nan
                    val = (p / (p_total + eps)) if cfg["use_relative"] else p
                    val = np.log10(val + eps) if cfg["use_log"] and np.isfinite(val) else val
                    row[f"{b}{name}"] = val.astype(np.float32)
        out.append(row)
    
    return pd.DataFrame(out).sort_values("sec").reset_index(drop=True)

def extract_1hz_features(
    df: pd.DataFrame,
    **kwargs
    ) -> pd.DataFrame:  
    """
    extract 1Hz features from data at 128 Hz resolution

    Parameters
    ----------
    df: pd.DataFrame
        raw data at 128 Hz resolution, including EEG, vital signs, and drugs
    fs: int
        Sampling frequency
    cols: Dict[str, Sequence[str]]
        Dictionary of column names and their corresponding data
        columns:
            - "bis": str
                Column name for BIS data
            - "emg": str
                Column name for EMG data
            - "sqi": str
                Column name for SQI data
            - "eeg": Sequence[str]
                Sequence of EEG channels
            - "vital": Sequence[str]
                Sequence of vital signs
            - "drug": Sequence[str]
                Sequence of drugs
    bis_ffill_seconds: int
        Number of seconds to fill BIS data
    sqi_threshold: float
        Threshold for SQI data
    lag_seconds: Optional[int]
        Number of seconds to lag BIS data
    use_relative: bool
        Whether to use relative features
    use_log: bool
        Whether to use log features
    """
    cfg = apply_kwargs(_DEFAULTS, **kwargs)
    if cfg["cols"] is None or \
        any(k not in cfg["cols"] for k in ("bis", "sqi", "eeg")):
        raise ValueError("cols={'bis':..., 'sqi':..., 'eeg':[...] } is required")

    fs = int(cfg["fs"])
    if fs is None or fs <= 0:
        raise ValueError("fs must be a positive integer")

    cols = cfg["cols"]
    bis, sqi = \
        str(cols["bis"]), str(cols["sqi"])
    emg = str(cols["emg"]) if "emg" in cols else None

    eeg = list(cols.get("eeg", []))
    vital = list(cols.get("vital", []))
    drug = list(cols.get("drug", []))

    df = df.copy()
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- reindex bis and trim head ---
    df = reindex_bis_and_trim_head(df, **cfg)

    if "sec" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df["sec"] = ((df.index - df.index[0]).total_seconds()).astype("int32")
        else:
            rel = (df.index.values - df.index[0]).astype("float64")
            df["sec"] = np.floor(rel / float(fs))
    df["sec"] = pd.to_numeric(df["sec"], errors="coerce").astype(np.int32)
    
    # --- BIS: forward fill, followed by backfill ---
    lim = max(0, int(cfg["bis_ffill_seconds"] * fs) - 1)
    df[bis] = df[bis].ffill(limit=lim).bfill(limit=fs-1)

    # --- fill vital signals with ffill -> bfill (no limit) ---
    vital_cols = [c for c in (vital or []) if c in df.columns]
    drug_cols = [c for c in (drug or []) if c in df.columns]
    fill_cols = vital_cols + drug_cols
    
    for col in fill_cols:
        df[col] = df[col].ffill().bfill()

    # --- aggregate data at 1Hz ---
    agg_cols = [bis, sqi] + (vital or []) + (drug or [])
    if emg:
        agg_cols.append(emg)
    agg_cols = [c for c in agg_cols if c in df.columns]

    sec_agg = df.groupby("sec")[agg_cols].mean().reset_index() \
        if len(agg_cols) > 0 else pd.DataFrame(columns=["sec"])
    
    # Apply SQI threshold filter after aggregation
    if sqi in sec_agg.columns:
        sec_agg = sec_agg.loc[sec_agg[sqi] > float(cfg["sqi_threshold"])]

    # --- extract EEG bandpower at 1Hz ---
    eeg_available = [e for e in eeg if e in df.columns]
    eeg_feat = eeg_bandpower_1hz(df[eeg_available], df["sec"].to_numpy(), **cfg) \
        if len(eeg_available) > 0 else pd.DataFrame(columns=["sec"])

    # --- merge and drop nan ---
    merged = pd.merge(sec_agg, eeg_feat, on="sec", how="inner").dropna()

    # --- calculate lag features ---
    L = int(cfg["lag_seconds"]) if cfg["lag_seconds"] is not None else 0
    if L > 0:
        merged["BIS_target"] = merged[bis].shift(-L)
        head_cut = max(cfg["bis_ffill_seconds"], L)
        # Handle NA values in sec column
        sec_min = merged["sec"].min()
        sec_max = merged["sec"].max()
        if pd.notna(sec_min) and pd.notna(sec_max):
            s0 = int(sec_min) + head_cut
            s1 = int(sec_max) - L
            merged = merged[(merged["sec"] >= s0) & (merged["sec"] <= s1)]
        merged = merged.dropna(subset=["BIS_target"])
    else:
        merged["BIS_target"] = merged[bis]
    
    merged = merged.dropna(axis=1, how="all")
    
    return merged.reset_index(drop=True)