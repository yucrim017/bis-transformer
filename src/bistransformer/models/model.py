import torch
import torch.nn as nn
from bistransformer.models.encoder import BISEncoder
from bistransformer.models.head import RegressionHead

class BisAttentionRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        max_len: int,
        d_model: int = 256,
        n_head: int = 8,
        attention_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        pos_dropout: float = 0.0,
        hidden_size: int = 128,
        target_mode: str = "scalar",
        pool: str = "mean"
    ):
        super().__init__()
        
        self.encoder = BISEncoder(
            d_in=d_in,
            max_len=max_len,
            d_model=d_model,
            n_head=n_head,
            attention_layers=attention_layers,
            d_ff=d_ff,
            dropout=dropout,
            pos_dropout=pos_dropout,
        )
        
        self.head = RegressionHead(
            d_model=d_model,
            hidden_size=hidden_size,
            target_mode=target_mode,
            pool=pool,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, causal: bool=True) -> torch.Tensor:
        # x: (B, T, d_in)
        x = self.encoder(x, causal=causal)
        x = self.head(x)
        return x