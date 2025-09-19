import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Reuse attention and positional encodings from the main model module
from .cellgru import MHA_SDPA, Pos2DCache


class ControllerBlock(nn.Module):
    """Perceiver-style cross-attention block operating on controller tokens.

    Queries: controller tokens [B,Q,C]
    Keys/Values: memory tokens [B,S,C] (grid or support)
    """
    def __init__(self, c: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(c)
        self.ln_kv = nn.LayerNorm(c)
        self.cross = MHA_SDPA(c, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, 4 * c), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * c, c)
        )

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        # q_tokens: [B,Q,C], kv_tokens: [B,S,C]
        q = self.ln_q(q_tokens)
        kv = self.ln_kv(kv_tokens)
        q = q_tokens + self.cross(q, kv)
        q = q + self.ff(q)
        return q


class TransformerController2D(nn.Module):
    """
    Lightweight controller that reads pooled/full grid tokens and support tokens
    via cross-attention from a small set of controller tokens, performs light
    self-attention over controller tokens, and emits control heads:
      - FiLM gamma/beta (per-channel)
    """
    def __init__(self,
                 C: int,
                 n_heads: int = 8,
                 depth: int = 4,
                 G: int = 16,
                 Gs: int = 8,
                 use_grid_pool: bool = True,
                 pool_stride: int = 2,
                 self_attn_layers: int = 2,
                 max_steps: int = 512,
                 dropout: float = 0.0,
                 max_planes: int = 32):
        super().__init__()
        self.C = C
        self.G = G
        self.Gs = Gs
        self.use_grid_pool = use_grid_pool
        self.pool = nn.AvgPool2d(pool_stride, pool_stride) if use_grid_pool and pool_stride > 1 else None
        # Mixture and non-FiLM/gate heads removed for simplicity

        self.ctrl_tokens = nn.Parameter(torch.randn(G, C) / math.sqrt(C))
        self.scratchpad_proj = nn.Linear(C, C) if Gs > 0 else None
        # Default steer scale for task vectors (can be overridden from trainer)
        self.steer_scale = 1.0

        self.grid_ln = nn.LayerNorm(C)
        self.sup_ln = nn.LayerNorm(C)

        self.blocks = nn.ModuleList([ControllerBlock(C, n_heads, dropout) for _ in range(max(depth, 1))])
        self.self_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=C, nhead=n_heads, dim_feedforward=4 * C,
                                       dropout=dropout, batch_first=True)
            for _ in range(max(self_attn_layers, 0))
        ])

        # Plane embedding for multi-plane encoding (sized to max_planes)
        self.plane_emb = nn.Embedding(int(max(1, max_planes)), C)
        self.num_planes = 1  # will be set by CellGRU
        self.max_planes = int(max(1, max_planes))

        # Heads (per-plane FiLM + readout + optional router)
        self.film_gamma = nn.Linear(C, C * self.max_planes)
        self.film_beta = nn.Linear(C, C * self.max_planes)
        self.readout_alpha = nn.Linear(C, self.max_planes)
        self.router = nn.Linear(C, self.max_planes * self.max_planes)

        # Step embedding
        self.step_emb = nn.Embedding(max_steps, C)

    def encode_multi(self, mem_bphwc: torch.Tensor) -> torch.Tensor:
        """Encode multi-plane memory: [B,P,H,W,C] -> [B, P*Hp*Wp, C] with plane tags."""
        B, P, H, W, C = mem_bphwc.shape
        x = mem_bphwc.permute(0, 1, 4, 2, 3)  # [B,P,C,H,W]
        if self.pool is not None:
            x = x.reshape(B * P, C, H, W)
            x = self.pool(x)  # [B*P,C,Hp,Wp]
            Hp, Wp = x.shape[-2:]
            x = x.view(B, P, C, Hp, Wp)
        else:
            Hp, Wp = H, W

        seq = x.permute(0, 1, 3, 4, 2).reshape(B, P * Hp * Wp, C)  # [B,S,C]
        pos = Pos2DCache.get(Hp, Wp, C, seq.device, seq.dtype)[None, :, :]
        pos = pos.repeat(1, P, 1)  # [1, P*Hp*Wp, C]

        pids = torch.arange(P, device=seq.device)
        ptag = self.plane_emb(pids)[:, None, :].repeat(1, Hp * Wp, 1).reshape(P * Hp * Wp, C)
        ptag = ptag[None, :, :].expand(B, -1, -1)

        return self.grid_ln(seq + pos + ptag)

    def forward(self,
                mem_bphwc: torch.Tensor,
                support_tokens: Optional[torch.Tensor],
                step_t: int,
                scratchpad: Optional[torch.Tensor] = None,
                task_vec: Optional[torch.Tensor] = None,        # [B,C]
                task_prefix: Optional[torch.Tensor] = None):     # [B,Gτ,C]
        # mem_bphwc may be [B,H,W,C] for backward-compat; normalize to [B,P,H,W,C]
        if mem_bphwc.dim() == 4:
            mem_bphwc = mem_bphwc.unsqueeze(1)
        B, P, H, W, C = mem_bphwc.shape
        grid_seq = self.encode_multi(mem_bphwc)  # [B,Sg,C]
        sup_seq = self.sup_ln(support_tokens) if support_tokens is not None else None

        # Controller queries (learned + optional scratchpad + task-prefix)
        ctrl = self.ctrl_tokens[None].expand(B, -1, -1)  # [B,G,C]
        if self.Gs > 0 and scratchpad is not None:
            sp = self.scratchpad_proj(scratchpad) if self.scratchpad_proj is not None else scratchpad
            ctrl = torch.cat([ctrl, sp], dim=1)  # [B,G+Gs,C]
        if task_prefix is not None:
            ctrl = torch.cat([ctrl, task_prefix], dim=1)  # add τ-prefix queries

        # Add step embedding
        step_idx = torch.tensor(min(step_t, self.step_emb.num_embeddings - 1), device=ctrl.device)
        ctrl = ctrl + self.step_emb(step_idx)[None, None, :]

        # Cross-attention into grid then support (Perceiver style)
        for blk in self.blocks:
            ctrl = blk(ctrl, grid_seq)
            if sup_seq is not None:
                ctrl = blk(ctrl, sup_seq)

        # Light self-attention over controller tokens
        for sab in self.self_blocks:
            ctrl = sab(ctrl)

        # Pool controller tokens to a single vector
        h = ctrl.mean(dim=1)  # [B,C]

        # Steering: static (legacy) or per-batch task_vec
        if task_vec is not None:
            h = h + float(getattr(self, 'steer_scale', 1.0)) * task_vec
        elif hasattr(self, "steer_vec") and (self.steer_vec is not None) and hasattr(self, "steer_scale") and (self.steer_scale != 0.0):
            sv = self.steer_vec.to(h.device, h.dtype).view(1, -1)
            h = h + self.steer_scale * sv

        Puse = getattr(self, 'num_planes', 1)
        Puse = int(Puse)
        out = {}
        # Per-plane FiLM
        g = self.film_gamma(h)[:, : (C * Puse)].view(B, Puse, C)[:, :, None, None, :]
        b = self.film_beta(h)[:, : (C * Puse)].view(B, Puse, C)[:, :, None, None, :]
        out['gamma_p'] = g  # [B,P,1,1,C]
        out['beta_p'] = b   # [B,P,1,1,C]
        # Readout mix over planes
        alpha = torch.softmax(self.readout_alpha(h)[:, :Puse], dim=-1)  # [B,P]
        out['alpha'] = alpha
        # Optional router P x P
        raw = self.router(h)[:, : (Puse * Puse)].view(B, Puse, Puse)
        out['router'] = torch.softmax(raw, dim=-1)
        
        # New scratchpad tokens = last Gs tokens (if any)
        sp_out = ctrl[:, -self.Gs:, :] if self.Gs > 0 else None
        return out, sp_out
