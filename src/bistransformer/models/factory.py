from __future__ import annotations
from typing import Any


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
                "d_model": 128,
                "n_head": 8,
                "n_layers": 6,
                ...
            },
            "head": {
                "d_model": 128,
                "n_head": 8,
                "n_layers": 6,
                ...
            }
        }
    """
    name = str(_get(model_cfg, "name", "bis_regressor")).lower()
    if name in ["bis_regressor", "bis_attention", "transformer"]:
        from .model import BisAttentionRegressor
    
        d_in = _get(model_cfg, "input_dim", default=None) 
        target_mode = _get(model_cfg, "target_mode", default="scalar")

        # encoder config
        max_len = _get(model_cfg, "encoder", "max_len", default=60*128)
        d_model = _get(model_cfg, "encoder", "d_model", default=128)
        n_head = _get(model_cfg, "encoder", "n_head", default=4)
        n_layers = _get(model_cfg, "encoder", "n_layers", default=2)
        d_ff = _get(model_cfg, "encoder", "d_ff", default=256)
        dropout = _get(model_cfg, "encoder", "dropout", default=0.1)
        pos_dropout = _get(model_cfg, "encoder", "pos_dropout", default=0.0)
        
        # head config
        head_hidden = _get(model_cfg, "head", "hidden_size", default=128)
        head_pool = _get(model_cfg, "head", "pool", default="mean").lower()

        m = BisAttentionRegressor(
            max_len=max_len,
            d_in=int(d_in),
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            pos_dropout=pos_dropout,
            head_hidden=head_hidden,
            target_mode=target_mode,
            pool=head_pool,
        )

        m.hparams = {
            "name": name,
                "input_dim": int(d_in),
                "target_mode": target_mode,
                "encoder": {
                    "max_len": max_len,
                    "d_model": d_model,
                    "n_head": n_head,
                    "n_layers": n_layers,
                    "d_ff": d_ff,
                    "dropout": dropout,
                    "pos_dropout": pos_dropout,
                },
                "head": {
                    "hidden_size": head_hidden,
                    "pool": head_pool,
                },
        }
        return m

    raise ValueError(f"Unknown model name: {name}. "
                     f"supported: ['bis_regressor', 'bis_attention', 'transformer']")