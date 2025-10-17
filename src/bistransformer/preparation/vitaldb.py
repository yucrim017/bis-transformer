from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import time

import numpy as np
import pandas as pd
import vitaldb as vdb

from bistransformer.utils.data import (
    extract_1hz_features,
    columns_for_features,
    save_case_npz,
)

class VitalDBProcessor:
    def __init__(
        self,
        tracks: List[str],
        output_dir: Path,
        fs: float = 128.0,
        verbose: bool = True,
    ):
        self.tracks = tracks
        self.output_dir = output_dir
        self.fs = fs
        self.verbose = verbose

    def find_cases(
        self,
        n_cases: Optional[int] = None
    ) -> List[int]:
        cids = vdb.find_cases(self.tracks)
        if n_cases:
            cids = cids[:n_cases]
        return cids

    def process_case(
        self,
        cid: int,
        cols: Dict[str, Any],
        params: Dict[str, Any],
        save_npz: bool = True,
        save_parquet: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a case and save the results to a file.

        Parameters
        ----------
        cid: int
            Case ID
        cols: Dict[str, Any]
            List of tracks to process
        params: Dict[str, Any]
            Dictionary of parameters passed to the preprocessing function
        save_npz: bool
            Whether to save as npz file
        save_parquet: bool
            Whether to save as parquet file

        Returns
        -------
        Dict[str, Any]
            Dictionary of case results
        """
        start_time = time.time()

        if self.verbose:
            print(f"[VitalDBProcessor] Case {cid} ... fetching VitalDB")
            
        vf = vdb.VitalFile(cid, self.tracks)
        df_raw = vf.to_pandas(self.tracks, 1/self.fs)
        if df_raw.empty:
            raise RuntimeError(f"  empty dataframe")
        
        # --- extract 1Hz features ---
        kwargs = dict(params)
        kwargs["cols"] = cols
        df_feat = extract_1hz_features(df_raw, **kwargs)

        if self.verbose:
            print(f"  df_raw shape: {df_raw.shape}, df_feat shape: {df_feat.shape}")

        if df_feat.empty or ("BIS_target" not in df_feat.columns):
            raise RuntimeError(f"  no features or target generated")

        # --- process: df_feat -> X, y, T ---
        bis_col = cols["bis"]
        feat_cols = columns_for_features(df_feat, bis_col=bis_col)
        df_feat = df_feat.sort_values("sec").reset_index(drop=True)

        T = df_feat["sec"].to_numpy(dtype=np.int32)
        y = df_feat["BIS_target"].to_numpy(dtype=np.float32)[None, :]
        X = df_feat[feat_cols].to_numpy(dtype=np.float32).T

        # --- save files ---
        out_dir = self.output_dir / f"case_{cid:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = dict(
            cid=cid,
            seconds=int(y.shape[1]),
            n_features=X.shape[0],
            tracks=self.tracks,
            features=list(feat_cols)
            )

        if save_npz:
            save_case_npz(out_dir / "features.npz", X=X, y=y, sec=T, META=meta)
        if save_parquet:
            df_feat.to_parquet(out_dir / "table.parquet", index=False)
        if self.verbose:
            print(f"[VitalDBProcessor] Case {cid} saved: secs={meta['seconds']}, feats={meta['n_features']}")

        return dict(
            cid=cid,
            seconds=meta["seconds"],
            n_features=meta["n_features"],
            save_dir=str(out_dir),
            has_npz=bool(save_npz),
            has_parquet=bool(save_parquet),
            processing_time=time.time() - start_time,
        )

    def process_batch(
        self,
        case_ids: List[int],
        cols: Dict[str, Any],
        params: Dict[str, Any],
        save_npz: bool = True,
        save_parquet: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process a batch of cases and return a manifest dataframe
        """
        rows = []
        for cid in case_ids:
            try:
                row = self.process_case(
                    cid,
                    cols=cols,
                    params=params,
                    save_npz=save_npz,
                    save_parquet=save_parquet,
                )
                rows.append(row)
            except Exception as e:
                if self.verbose:
                    print(f"[VitalDBProcessor] Failed case {cid}: {e}")
                continue
        return pd.DataFrame(rows)
    
    def save_manifest(self, results: pd.DataFrame) -> Path:
        """
        Save the manifest dataframe to a file
        """
        path = self.output_dir / "manifest.csv"
        results.to_csv(path, index=False)
        if self.verbose:
            print(f"[VitalDBProcessor] Saved manifest: {path}")
        return path