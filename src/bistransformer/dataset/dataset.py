from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from bistransformer.utils.data import load_npz_case


def build_tracks_from_cfg(cfg_data) -> List[str]:
    """build tracks from config column names"""
    all_tracks = []
    for k in ["bis", "sqi", "emg", "eeg", "vital", "drug"]:
        val = cfg_data.cols.get(k, [])
        if val is None:
            continue
        if isinstance(val, str):
            all_tracks.append(val)
        else:
            # Handle lists and OmegaConf ListConfig
            try:
                all_tracks.extend(list(val))
            except (TypeError, ValueError):
                pass
    return sorted(set(all_tracks))


def build_tracks_from_dict(cols_dict: Dict) -> List[str]:
    """build tracks from columns dictionary"""
    all_tracks = []
    for k in ["bis", "sqi", "emg", "eeg", "vital", "drug"]:
        val = cols_dict.get(k)
        if val is None:
            continue
        # Handle empty lists/tuples
        try:
            if len(val) == 0:
                continue
        except TypeError:
            # val is not iterable (e.g., a string), continue processing
            pass
        
        if isinstance(val, str):
            all_tracks.append(val)
        elif isinstance(val, (list, tuple)):
            all_tracks.extend(val)
        else:
            # Handle OmegaConf ListConfig or other iterables
            try:
                all_tracks.extend(list(val))
            except (TypeError, ValueError):
                pass
    return sorted(set(all_tracks))


def _sliding_window(
    n: int,
    win_sec: int,
    hop_sec: int
) -> List[int]:
    """
    sliding window

    Parameters
    ----------
    n: int
        Length of the sequence
    win_sec: int
        Window size
    hop_sec: int
        Hop size

    Returns
    -------
    List[int]
        List of indices
    """
    if n < win_sec:
        return []
    return list(range(0, n - win_sec + 1, hop_sec))


@dataclass
class WindowConfig:
    win_sec: int = 60
    hop_sec: int = 4
    target: str = "scalar" # "scalar" | "sequence"
    target_reduce: str = "last" # when target="scalar", "last" | "mean" | "median"

@dataclass
class NormConfig:
    """normalize features for each case"""
    enable: bool = False
    method: str = "zscore" # "zscore" | "minmax"
    eps: float = 1e-6


class BISDataset(Dataset):
    f"""
    load npz case and expand data by sliding window
     - X: (n_features, n_seconds) -> (window_size, n_windows, n_features)
     - y: scalar -> (window_size, n_windows)

    Parameters
    ----------
    case_dirs: Path
        Path to the case directory
    win_sec: int
        Window size
    hop_sec: int
        Hop size

    Returns
    -------
    Dict[str, Any]
        Dictionary of data:
         - inputs: (win, 1, D) float32
         - targets: (win, ) float32
         - cid: int
         - t0: int, time of the first window
         - sec: (win, ) int, length of the sequence
    """
    def __init__(
        self,
        case_dirs: List,
        win_cfg: WindowConfig,
        norm_cfg: Optional[NormConfig] = None
    ) -> None:
        self.case_dirs = case_dirs
        self.win = win_cfg.win_sec
        self.hop = win_cfg.hop_sec
        self.target_mode = str(win_cfg.target)
        self.target_reduce = str(win_cfg.target_reduce)
        self.norm = norm_cfg or NormConfig(False)

        self._cases: List[Dict[str, Any]] = []
        self._index: List[Tuple[int, int]] = []

        for ci, d in enumerate(self.case_dirs):
            try:
                data = load_npz_case(d)
            except Exception as e:
                print(f"[BISDataset] error loading case {d}: {e}")
                continue

            X, y, sec = data["X"], data["y"], data["sec"]

            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            sec = np.asarray(sec, dtype=np.int32)

            T = X.shape[1]
            starts = _sliding_window(T, self.win, self.hop)
            if not starts:
                continue

            if self.norm.enable:
                mu = X.mean(axis=1)
                std = X.std(axis=1) + self.norm.eps
                X = (X - mu[:, None]) / std[:, None]

            self._cases.append(dict(
                X=X,
                y=y,
                sec=sec,
                meta=data.get("meta", {}),
                T=T,
            ))
            for st in starts:
                self._index.append((ci, st))

        if len(self._index) == 0:
            raise RuntimeError("No window samples constructed. Check win/hop and data length.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ci, st = self._index[i]
        c = self._cases[ci]
        X = c["X"][:, st:st+self.win] # (D, win)
        y = c["y"][:, st:st+self.win]  # (1, win)
        
        sec = c["sec"][st:st+self.win] # (win,)

        # transpose X to (T, D)
        x_t = X.T

        # target
        if self.target_mode == "sequence": # (T, 1)
            tgt = y.T  # (win, 1)
        else: # scalar
            if self.target_reduce == "mean":
                v = np.nanmean(y)
            elif self.target_reduce == "median":
                v = np.nanmedian(y)
            else: # "last"
                v = y[0, -1]  # (1, win) -> scalar
            tgt = np.array([v], dtype=np.float32)  # (1,)

        meta = c.get("meta", {})
        if not isinstance(meta, dict):
            if isinstance(meta, np.ndarray) and meta.dtype == object:
                try:
                    import json
                    meta = json.loads(meta.item())
                except Exception:
                    meta = {}
            else:
                meta = {}
        cid = int(meta.get("cid", self._cid_from_path(self.case_dirs[ci])))

        return dict(
            inputs=torch.from_numpy(x_t).float(),   # (T, D)
            targets=torch.from_numpy(tgt).float(),  # (1,) or (T,)
            cid=cid,
            t0=int(st),
            sec=torch.from_numpy(sec.astype(np.int32)), # (T,)
        )
    
    def _cid_from_path(self, p: Path) -> int:
        """get cid from path (expecting path/to/case_000123)"""
        try:
            name = p.name
            if name.startswith("case_"):
                return int(name.split("_")[1])
        except Exception:
            pass
        return -1