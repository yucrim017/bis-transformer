import torch
import torch.nn as nn
from bistransformer.models.layers import (
    TransformerEncoderLayer,
    SinusoidalPositionalEncoding
    )


def _causal_mask(T: int) -> torch.Tensor:
    mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool), 
        diagonal=1)
    return mask

class BISEncoder(nn.Module):
    def __init__(
        self, 
        d_in: int, 
        d_model: int=128, 
        n_head: int=4,
        n_layers: int=4,
        d_ff: int=256, 
        max_len: int=60*128,
        dropout: float=0.1,
        pos_dropout: float=0.0,
        batch_first: bool=True
        ):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len, pos_dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal: bool=False) -> torch.Tensor:
        # x: (B, T, d_in)
        x = self.embed(x)
        x = self.pos(x)
        attn_mask = _causal_mask(x.size(1)) if causal else None
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
            x = self.norm(x)
        return x