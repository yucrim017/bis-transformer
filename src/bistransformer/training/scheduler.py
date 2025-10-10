import torch

def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    name = getattr(cfg.train.scheduler, "name", "cosine").lower()
    epochs = int(getattr(cfg.train, "epochs", 50))
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(getattr(cfg.train.scheduler, "min_lr", 1e-6))
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(getattr(cfg.train.scheduler, "mode", "min")),
            factor=float(getattr(cfg.train.scheduler, "factor", 0.1)),
            patience=int(getattr(cfg.train.scheduler, "lr_patience", 3))
        )
    else:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _: float(getattr(cfg.train.scheduler, "lr_lambda", 1.0))
        )