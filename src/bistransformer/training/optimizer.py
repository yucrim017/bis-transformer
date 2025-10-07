import torch

def build_optimizer(model: torch.nn.Module, cfg):
    name = getattr(cfg.train.optimizer, "name", "adamw").lower()
    lr = float(getattr(cfg.train.optimizer, "lr", 3e-4))
    wd = float(getattr(cfg.train.optimizer, "weight_decay", 1e-2))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        mom = float(getattr(cfg.train.optimizer, "momentum", 0.9))
        nesterov = getattr(cfg.train.optimizer, "nesterov", True)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd, nesterov=nesterov)
    else: # adamw
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)