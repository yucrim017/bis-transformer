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
        self.fc = nn.Linear(d_model, hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) -> scalar:(B, 1), sequence:(B, T, 1)

        if self.target_mode == "scalar": # (B, T, d_model) -> (B, d_model)
            # scalar head: pooling
            if self.pool == 'mean':
                x = x.mean(dim=1)
            elif self.pool == 'max':
                x = x.max(dim=1)[0]
            elif self.pool == 'cls':
                x = x[:, 0, :]
            elif self.pool == 'last':
                x = x[:, -1, :]
            else:
                raise ValueError(f"Invalid pool type: {self.pool}")
        elif self.target_mode == "sequence":
            pass
        else:
            raise ValueError(f"Invalid target mode: {self.target_mode}")

        x = self.fc(self.act(x))
        x = self.fc2(self.dropout(x))
        
        # BIS value is limited to 0-100 range
        x = torch.sigmoid(x) * 100.0
        
        return x

