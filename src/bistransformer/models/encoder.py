import torch
import torch.nn as nn
from bistransformer.models.block import (
    TransformerEncoderLayer,
    SinusoidalPositionalEncoding
    )


def _causal_mask(T: int, device=None) -> torch.Tensor:
    """Create causal mask on specified device"""
    mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device), 
        diagonal=1)
    return mask

class BISEncoder(nn.Module):
    def __init__(
        self, 
        d_in: int, 
        max_len: int,
        d_model: int=256, 
        n_head: int=8,
        attention_layers: int=4,
        d_ff: int=256, 
        dropout: float=0.1,
        pos_dropout: float=0.0,
        batch_first: bool=True
        ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(d_in, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        self.pos = SinusoidalPositionalEncoding(d_model, max_len, pos_dropout)
        self.attention_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(attention_layers)
        ])

    def forward(self, x: torch.Tensor, causal: bool=True) -> torch.Tensor:
        # x: (B, T, d_in)
        x = self.embed(x)
        x = self.pos(x)
        attn_mask = _causal_mask(x.size(1), device=x.device) if causal else None
        for layer in self.attention_layers:
            x = layer(x, attn_mask=attn_mask)
        return x