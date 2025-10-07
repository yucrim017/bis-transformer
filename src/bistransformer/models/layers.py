import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        max_len: int=60*128, 
        dropout: float=0.0,
        batch_first: bool=True
        ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() \
                                * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int=256, 
        dropout: float=0.1,
        batch_first: bool=True
        ):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        dropout: float=0.1,
        batch_first: bool=True
        ):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor=None) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, _ = x.size()
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(B, T, self.n_head, self.d_k).permute(2, 0, 1, 3) # (n_head, B, T, d_k)
        k = k.view(B, T, self.n_head, self.d_k).permute(2, 0, 1, 3)
        v = v.view(B, T, self.n_head, self.d_k).permute(2, 0, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e9)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn) # (n_head, B, T, T)
        x = torch.matmul(attn, v) # (n_head, B, T, d_k)
        x = x.permute(1, 2, 0, 3).contiguous()
        x = x.view(B, T, self.d_model)
        x = self.w_o(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        d_ff: int=256, 
        dropout: float=0.1,
        batch_first: bool=True
        ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor=None) -> torch.Tensor:
        # x: (B, T, d_model)
        x = self.norm1(
            x + self.dropout(self.self_attn(x, attn_mask=attn_mask))
            )
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x