import torch
import torch.nn as nn
from typing import Optional


def _init_vec(d: int) -> torch.Tensor:
    v = torch.zeros(d)
    nn.init.normal_(v, std=0.02)
    return v


class TaskLatentBank(nn.Module):
    """
    Per-task code τ_t plus light projections into model spaces.
    We keep a ParameterDict so tasks can be pre-created from the trainer.
    """
    def __init__(self, dim: int, C: int, prefix_tokens: int = 0):
        super().__init__()
        self.dim = int(dim)
        self.C = int(C)
        self.codes = nn.ParameterDict()

        # Projections
        self.to_ctrl_steer = nn.Linear(dim, C)                   # τ -> steer vector in controller space
        self.to_ctrl_prefix = nn.Linear(dim, prefix_tokens * C)  # τ -> Gτ prefix tokens
        self.to_film = nn.Linear(dim, 2 * C)                     # τ -> [γ, β] in feature space

        self.prefix_tokens = int(prefix_tokens)

    @staticmethod
    def normalize_key(task_id: str) -> str:
        return task_id.replace("/", "_").replace(":", "_")

    def ensure(self, task_id: str):
        key = self.normalize_key(task_id)
        if key not in self.codes:
            self.codes[key] = nn.Parameter(_init_vec(self.dim))

    def get(self, task_id: str, device=None, dtype=None) -> torch.Tensor:
        key = self.normalize_key(task_id)
        z = self.codes[key]
        if device is not None or dtype is not None:
            z = z.to(device=device, dtype=dtype)
        return z

    # Projections (batched)
    def ctrl_steer(self, z_b: torch.Tensor) -> torch.Tensor:
        # z_b: [B, dim] -> [B, C]
        return self.to_ctrl_steer(z_b)

    def ctrl_prefix(self, z_b: torch.Tensor) -> Optional[torch.Tensor]:
        # -> [B, Gτ, C]
        if self.prefix_tokens <= 0:
            return None
        t = self.to_ctrl_prefix(z_b).view(z_b.size(0), self.prefix_tokens, self.C)
        return t

    def film(self, z_b: torch.Tensor):
        # -> γ, β each [B, 1, 1, 1, C]
        gb = self.to_film(z_b)
        g, b = gb.chunk(2, dim=-1)
        return g[:, None, None, None, :], b[:, None, None, None, :]

