import os
from dataclasses import dataclass

@dataclass
class EarlyStopping:
    monitor: str = "val/mae"
    patience: int = 8
    best: float = float("inf")
    count: int = 0

    def step(self, value: float) -> bool:
        if value < self.best - 1e-9:
            self.best = value
            self.count = 0
            return False
        self.count += 1
        return self.count > self.patience

class CheckpointSaver:
    def __init__(
        self,
        dirpath: str = "outputs/checkpoints",
        filename: str = "best.pt"
    ):
        self.dirpath = dirpath
        self.filename = filename
        os.makedirs(dirpath, exist_ok=True)
        self.best_path = os.path.join(self.dirpath, self.filename)

    def save(self, model_state_dict: dict):
        import torch
        torch.save(model_state_dict, self.best_path)
        return self.best_path