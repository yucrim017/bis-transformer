import torch

def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.mean(torch.abs(y_pred - y_true)).item()

def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

def pearson(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    x = y_pred.view(-1) - torch.mean(y_pred)
    y = y_true.view(-1) - torch.mean(y_true)
    denom = torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2))
    if denom.item() == 0:
        return 0.0
    return (torch.sum(x * y) / denom).item()