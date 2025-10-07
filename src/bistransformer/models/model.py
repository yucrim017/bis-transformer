import torch
import torch.nn as nn
from bistransformer.models.encoder import BISEncoder
from bistransformer.models.heads import RegressionHead


class BisAttentionRegressor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = BISEncoder(**kwargs)
        self.head = RegressionHead(**kwargs)

    def forward(self, x: torch.Tensor, causal: bool=False) -> torch.Tensor:
        # x: (B, T, d_in)
        x = self.encoder(x, causal=causal)
        x = self.head(x)
        return x