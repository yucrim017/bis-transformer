from __future__ import annotations
from typing import Any

from .model import BisAttentionRegressor


def _get(obj: Any, *keys, default=None):
    """utility function to get nested attributes from an object"""
    cur = obj
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            cur = getattr(cur, k, None)
    return default if cur is None else cur


def build_model(model_cfg):
    """
    build model from model config
    
    example:
        model_cfg = {
            "encoder": {
                "d_model": 256,
                "n_head": 8,
                "attention_layers": 4,
                ...
            },
            "head": {
                "d_model": 256,
                "n_head": 8,
                "attention_layers": 4,
                ...
            }
        }
    """
    name = str(_get(model_cfg, "name", default="bis_regressor")).lower()
    if name in ["bis_regressor", "bis_attention", "transformer"]:
    
        d_in = _get(model_cfg, "d_in", default=None) 
        max_len = _get(model_cfg, "max_len", default=None)
        target_mode = _get(model_cfg, "target_mode", default="scalar")

        # encoder config
        d_model = _get(model_cfg, "encoder", "d_model", default=256)
        n_head = _get(model_cfg, "encoder", "n_head", default=8)
        attention_layers = _get(model_cfg, "encoder", "attention_layers", default=4)
        d_ff = _get(model_cfg, "encoder", "d_ff", default=256)
        dropout = _get(model_cfg, "encoder", "dropout", default=0.1)
        pos_dropout = _get(model_cfg, "encoder", "pos_dropout", default=0.0)
        
        # head config
        head_hidden_size = _get(model_cfg, "head", "hidden_size", default=256)
        head_pool = _get(model_cfg, "head", "pool", default="mean").lower()

        m = BisAttentionRegressor(
            d_in=int(d_in),
            max_len=max_len,
            d_model=d_model,
            n_head=n_head,
            attention_layers=attention_layers,
            d_ff=d_ff,
            dropout=dropout,
            pos_dropout=pos_dropout,
            hidden_size=head_hidden_size,
            target_mode=target_mode,
            pool=head_pool
        )

        m.hparams = {
            "name": name,
                "d_in": int(d_in),
                "max_len": max_len,
                "target_mode": target_mode,
                "encoder": {
                    "d_model": d_model,
                    "n_head": n_head,
                    "attention_layers": attention_layers,
                    "d_ff": d_ff,
                    "dropout": dropout,
                    "pos_dropout": pos_dropout,
                },
                "head": {
                    "hidden_size": head_hidden_size,
                    "pool": head_pool,
                },
        }
        return m

    raise ValueError(f"Unknown model name: {name}. "
                     f"supported: ['bis_regressor', 'bis_attention', 'transformer']")