# src/models/adapters.py
import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 0, alpha: float = 1.0, dropout_p: float = 0.0):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.scale = alpha / max(1, r)
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        if r > 0:
            self.A = nn.Linear(base.in_features, r, bias=False)
            self.B = nn.Linear(r, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
            nn.init.zeros_(self.B.weight)
        else:
            self.A = self.B = None

    def forward(self, x):
        y = self.base(x)
        if self.r > 0:
            y = y + self.scale * self.B(self.A(self.drop(x)))
        return y

def wrap_linear_with_lora(module: nn.Module, r: int, alpha: float):
    """Recursively wraps all nn.Linear with LoRA; returns number wrapped."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            n += 1
        else:
            n += wrap_linear_with_lora(child, r, alpha)
    return n
