from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random

import yaml
from torch.utils.data import DataLoader

from bistransformer.dataset import (
    BISDataset, 
    WindowConfig, 
    NormConfig, 
    build_tracks_from_dict
)


def _read_splits_yaml(path: Path) -> Dict[str, List[int]]:
    """
    read splits from yaml file

    example:
        train: [1, 2, 3]
        val: [4, 5, 6]
        test: [7, 8, 9]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: list(v or []) for k, v in data.items()}

def _discover_all_cases(processed_root: Path) -> List[int]:
    """
    Discover all case IDs from processed_root directory
    
    Returns:
        List of case IDs
    """
    case_dirs = sorted(processed_root.glob("case_*"))
    cids = []
    for d in case_dirs:
        if (d / "features.npz").exists():
            try:
                # Extract cid from "case_000123" format
                cid = int(d.name.split("_")[1])
                cids.append(cid)
            except (IndexError, ValueError):
                continue
    return sorted(cids)

def _auto_split_cases(
    cids: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[int]]:
    """
    Automatically split cases into train/val/test
    
    Parameters
    ----------
    cids: List[int]
        List of all case IDs
    train_ratio: float
        Ratio of training set
    val_ratio: float
        Ratio of validation set
    test_ratio: float
        Ratio of test set
    seed: int
        Random seed for reproducibility
    
    Returns
    -------
    Dict[str, List[int]]
        Dictionary with 'train', 'val', 'test' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Shuffle with fixed seed
    random.seed(seed)
    cids_shuffled = cids.copy()
    random.shuffle(cids_shuffled)
    
    n = len(cids_shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_cids = cids_shuffled[:n_train]
    val_cids = cids_shuffled[n_train:n_train + n_val]
    test_cids = cids_shuffled[n_train + n_val:]
    
    return {
        "train": sorted(train_cids),
        "val": sorted(val_cids),
        "test": sorted(test_cids)
    }

def _case_dirs_from_cids(
    processed_root: Path,
    cids: List[int],
    cfg_cols: Optional[Dict] = None
) -> List[Path]:
    """list case directories from cids, validate tracks with cfg.data.cols"""
    import numpy as np
    import json
    
    # Build expected tracks from cfg.data.cols
    expected_tracks = None
    if cfg_cols:
        expected_tracks = build_tracks_from_dict(cfg_cols)
        if expected_tracks:
            print(f"[datamodule] Expected tracks: {expected_tracks}")
    
    dirs = []
    skipped_count = 0
    for cid in cids:
        d = processed_root / f"case_{cid:06d}"
        if not (d / "features.npz").exists():
            continue
            
        # Validate tracks if cfg_cols provided
        if expected_tracks is not None:
            try:
                f = np.load(d / "features.npz", allow_pickle=True)
                
                # Load meta and extract tracks
                if "META" in f.files:
                    meta_data = f["META"]
                    if isinstance(meta_data, np.ndarray) and meta_data.dtype == object:
                        meta = json.loads(meta_data.item())
                    elif isinstance(meta_data, np.ndarray):
                        meta = meta_data.item()
                    else:
                        meta = {}
                    
                    case_tracks = sorted(meta.get("tracks", []))
                    f.close()
                    
                    # Check if tracks match
                    if case_tracks != expected_tracks:
                        print(f"[datamodule] Skipping case {cid}: tracks mismatch")
                        skipped_count += 1
                        continue
                else:
                    f.close()
                    print(f"[datamodule] Skipping case {cid}: no META found")
                    skipped_count += 1
                    continue
                    
            except Exception as e:
                print(f"[datamodule] Warning: Could not validate case {cid}: {e}")
                skipped_count += 1
                continue
        
        dirs.append(d)
    
    if skipped_count > 0:
        print(f"[datamodule] Skipped {skipped_count} cases due to track mismatch")
    
    return dirs

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    build dataloaders from Hydra config

    example:
        cfg.data.paths.processed: "data/processed/v1"
        cfg.data.paths.splits:    "configs/data/splits.yaml"  # optional
        cfg.data.auto_split.enable: true  # if true, auto-split cases
        cfg.data.auto_split.train_ratio: 0.8
        cfg.data.auto_split.val_ratio: 0.1
        cfg.data.auto_split.test_ratio: 0.1
        cfg.data.auto_split.seed: 42
        cfg.data.window.length: 60
        cfg.data.window.hop:     4
        cfg.data.loader.batch_size: 32
        cfg.data.loader.num_workers: 4
        cfg.data.loader.pin_memory: true
        cfg.data.loader.drop_last: false
        cfg.data.normalization.enable: true
        cfg.data.normalization.eps: 1e-6
    """
    processed_root = Path(cfg.data.paths.processed)
    
    # Determine splits: auto-split or from yaml
    auto_split = getattr(cfg.data, "auto_split", None)
    if auto_split and getattr(auto_split, "enable", False):
        # Auto-split mode
        print(f"[datamodule] Auto-splitting cases from {processed_root}")
        all_cids = _discover_all_cases(processed_root)
        print(f"[datamodule] Found {len(all_cids)} cases")
        
        splits = _auto_split_cases(
            all_cids,
            train_ratio=float(getattr(auto_split, "train_ratio", 0.8)),
            val_ratio=float(getattr(auto_split, "val_ratio", 0.1)),
            test_ratio=float(getattr(auto_split, "test_ratio", 0.1)),
            seed=int(getattr(auto_split, "seed", 42))
        )
        print(f"[datamodule] Split: train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")
    else:
        # Read from splits yaml
        splits_path = Path(cfg.data.paths.splits)
        if not splits_path.exists():
            raise FileNotFoundError(
                f"Splits file not found: {splits_path}\n"
                f"Set cfg.data.auto_split.enable=true to auto-split cases"
            )
        print(f"[datamodule] Loading splits from {splits_path}")
        splits = _read_splits_yaml(splits_path)
    
    train_cids = splits.get("train", [])
    val_cids = splits.get("val", [])
    test_cids = splits.get("test", [])
    
    # list case directories with track validation
    cfg_cols = dict(cfg.data.cols) if hasattr(cfg.data, 'cols') else None
    train_dirs = _case_dirs_from_cids(processed_root, train_cids, cfg_cols)
    val_dirs = _case_dirs_from_cids(processed_root, val_cids, cfg_cols)
    test_dirs = _case_dirs_from_cids(processed_root, test_cids, cfg_cols)

    # set window and normalization config
    win_cfg = WindowConfig(
        win_sec=cfg.data.window.length,
        hop_sec=cfg.data.window.hop,
        target=str(getattr(cfg.model, "target_mode", "scalar")) \
            if hasattr(cfg.model, "target_mode") else "scalar",
        target_reduce=str(getattr(cfg.model, "target_reduce", "last")) \
            if hasattr(cfg.model, "target_reduce") else "last",
    )
    norm_cfg = NormConfig(
        enable=bool(getattr(cfg.data.normalization, "enable", False)),
        eps=getattr(cfg.data.normalization, "eps", 1e-6),
    )

    # build datasets
    train_ds = BISDataset(train_dirs, win_cfg, norm_cfg)
    val_ds = BISDataset(val_dirs, win_cfg, norm_cfg)
    test_ds = BISDataset(test_dirs, win_cfg, norm_cfg)

    # common dataloader settings
    common = dict(
        batch_size=int(cfg.data.loader.batch_size),
        num_workers=int(cfg.data.loader.num_workers),
        pin_memory=bool(cfg.data.loader.pin_memory),
        drop_last=bool(cfg.data.loader.drop_last),
    )

    # build dataloaders
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader
