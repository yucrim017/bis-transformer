import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int=128,
        target_mode: str="scalar",
        pool='max',
        dropout: float=0.1,
        batch_first: bool=True
        ):
        super().__init__()
        self.target_mode = target_mode
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = self.norm(x)

        if self.target_mode == "sequence":
            # sequence head: no pooling
            x = self.fc(self.act(x)) # (B, T, hidden_size)
            x = self.fc2(self.dropout(x))
            return x # (B, T, 1)
        else:
            # scalar head: pooling
            if self.pool == 'mean':
                x = x.mean(dim=1) # (B, hidden_size)
            elif self.pool == 'max':
                x = x.max(dim=1)[0]
            elif self.pool == 'cls':
                x = x[:, 0, :]
            else:
                raise ValueError(f"Invalid pool type: {self.pool}")
            x = self.fc(self.act(x))
            x = self.fc2(self.dropout(x))
            return x # (B, 1)

