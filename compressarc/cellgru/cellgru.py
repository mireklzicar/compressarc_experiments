# src/models/cellgru.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint as _ckpt
from typing import List, Optional, Tuple
from .ccp import CCP
from .task_latent import TaskLatentBank


class Pos2DCache:
    _cache = {}

    @staticmethod
    def get(H: int, W: int, C: int, device: torch.device, dtype: torch.dtype):
        """Return 2D sinusoidal positions [H*W, C]. When compiling, avoid Python
        dict caching to keep the region in-graph.
        """
        # Prefer torch.compiler.is_compiling if available, else fall back to torch._dynamo
        try:
            is_comp = getattr(torch.compiler, "is_compiling", None)
        except Exception:
            is_comp = None
        if is_comp is None:
            try:
                import torch._dynamo as _dynamo  # type: ignore
                dyn_comp = getattr(_dynamo, "is_compiling", lambda: False)
                compiling = bool(dyn_comp())
            except Exception:
                compiling = False
        else:
            compiling = bool(is_comp())

        if compiling:
            return GlobalAttnGRU2D._pos_2d(H, W, C).to(device=device, dtype=dtype)

        key = (H, W, C, device.type, str(dtype))
        if key not in Pos2DCache._cache:
            pos = GlobalAttnGRU2D._pos_2d(H, W, C).to(device=device, dtype=dtype)
            Pos2DCache._cache[key] = pos
        return Pos2DCache._cache[key]

__all__ = ["CellGRU", "DCGRU2D"]


class MHA_SDPA(nn.Module):
    """
    Multi-head attention using PyTorch scaled_dot_product_attention (Flash/SDPA).
    Expects inputs in [B, S, C] and returns [B, S, C].
    """
    def __init__(self, c: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert c % num_heads == 0
        self.h = num_heads
        self.d = c // num_heads
        self.q_proj = nn.Linear(c, c, bias=True)
        self.k_proj = nn.Linear(c, c, bias=True)
        self.v_proj = nn.Linear(c, c, bias=True)
        self.out_proj = nn.Linear(c, c, bias=True)
        self.dropout_p = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,S,C] -> [B,h,S,d]
        B, S, C = x.shape
        return x.view(B, S, self.h, self.d).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,h,S,d] -> [B,S,C]
        B, h, S, d = x.shape
        return x.transpose(1, 2).reshape(B, S, h * d)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        # x_*: [B, S, C]
        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )  # [B,h,S,d]
        out = self._merge_heads(attn)  # [B,S,C]
        return self.out_proj(out)

class CellGRU(nn.Module):
    def __init__(self, num_units, n_input, n_classes, dropout_keep_prob=1.0,
                 kernel_size=3, neighborhood="von_neumann", use_diagonals=False,
                 cell_type="dcgru",  # "dcgru"
                 attn_heads=8, num_latents=64, toroidal=False,
                 conditioning: str = "none",  # "none" | "film" | "prompt2d" | "controller"
                 prompt_hmax: int = 30, prompt_wmax: int = 30,
                 # Task latent τ options
                 task_latent_dim: int = 0,
                 task_latent_tokens: int = 0,
                 task_latent_use_film: bool = False,
                 # Controller config
                 controller_depth: int = 4,
                 controller_heads: int = 8,
                 controller_tokens: int = 16,
                 controller_scratch: int = 8,
                 controller_pool_stride: int = 2,
                 # prompt/latents/mix removed from controller path
                 controller_update_every: int = 1,
                 controller_enable_film: bool = True,
                 num_planes: int = 1,
                 # Global mixer knobs (only 'full' or 'none' supported)
                 mixer_type: str = "full",  # "none" | "full"
                 mixer_every: int = 4,
                 mixer_heads: int = 8,
                 mixer_depth: int = 1,
                 mixer_pool_stride: int = 4,
                 # Force one plane to hold input features at every step
                 force_input_plane: bool = False,
                 input_plane_index: int = 0):
        super().__init__()
        self.n_classes = n_classes
        self.n_input = n_input
        self.num_units = num_units
        self.dropout_prob = 1.0 - dropout_keep_prob
        if use_diagonals:
            neighborhood = "moore"
        self.neighborhood = neighborhood
        self.toroidal = toroidal

        self.input_layer = nn.Conv2d(n_input, num_units, kernel_size=1)

        # Conditioning configuration
        assert conditioning in ("none", "film", "prompt2d", "controller")
        self.conditioning = conditioning
        self.num_latents = num_latents
        self.attn_heads = attn_heads
        self.prompt_hmax = prompt_hmax
        self.prompt_wmax = prompt_wmax
        # No global token state (global_tokens cell removed)
        # Controller knobs
        self.controller_update_every = max(1, int(controller_update_every))
        self.controller_enable_film = bool(controller_enable_film)
        # Only FiLM head is supported in controller mode

        # Task latent configuration
        self.task_latent_dim = int(task_latent_dim)
        self.task_latent_tokens = int(task_latent_tokens)
        self.task_latent_use_film = bool(task_latent_use_film)
        self.active_task_id: Optional[str] = None
        if self.task_latent_dim > 0:
            self.task_bank = TaskLatentBank(dim=self.task_latent_dim, C=num_units,
                                            prefix_tokens=self.task_latent_tokens)
        else:
            self.task_bank = None

        if cell_type == "dcgru":
            self.cell = DCGRU2D(num_units, kernel_size=kernel_size,
                                dropout=self.dropout_prob, neighborhood=neighborhood,
                                toroidal=self.toroidal)
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")

        self.output_layer = nn.Conv2d(num_units, n_classes, kernel_size=1)

        # Multi-plane scratchpad configuration (fixed to 1)
        self.num_planes = 1

        # Optional 2D-aware global mixer
        self.mixer_every = max(1, int(mixer_every))
        self.mixer = None
        if mixer_type and mixer_type != "none":
            if mixer_type == "full":
                self.mixer = GlobalMixer2D(C=num_units, heads=mixer_heads, depth=mixer_depth, dropout=self.dropout_prob)
            else:
                raise ValueError(f"Unknown mixer_type: {mixer_type}")

        self.saturation_limit = 0.9
        self.saturation_costs = []
        self.force_input_plane = bool(force_input_plane)
        self.input_plane_index = int(max(0, input_plane_index))
        # Compiled step hook (set by regional compilation utility if enabled)
        self._compiled_step = None

        # Memory/compute control knobs (can be overridden from train.py)
        self.use_checkpoint: bool = False  # activation checkpointing per recurrent step
        self.detach_every: Optional[int] = None  # detach state every N steps in forward()
        self.sat_every: int = 1  # only record saturation loss every N steps

        # Optional conditioning modules
        if self.conditioning in ("film", "prompt2d"):
            self.program_encoder = ProgramEncoder(self.input_layer, c=num_units)
        if self.conditioning == "film":
            # Map z -> per-channel gamma/beta
            self.film_gamma = nn.Linear(num_units, num_units)
            self.film_beta  = nn.Linear(num_units, num_units)
        if self.conditioning == "prompt2d":
            self.prompt2d = Prompt2D(C=num_units, Hmax=prompt_hmax, Wmax=prompt_wmax,
                                     program_encoder=self.program_encoder)
        if self.conditioning in ("controller"):
            # Build support->latents generator for latent RW cell usage
            self.support_to_latents = SupportLatentGenerator(
                input_layer=self.input_layer,
                C=num_units,
                num_latents=num_latents,
                num_heads=attn_heads
            )
        if self.conditioning == "controller":
            from .controller import TransformerController2D
            self.controller = TransformerController2D(
                C=num_units,
                n_heads=controller_heads,
                depth=controller_depth,
                G=controller_tokens,
                Gs=controller_scratch,
                use_grid_pool=(controller_pool_stride > 1),
                pool_stride=controller_pool_stride,
                self_attn_layers=2,
                dropout=self.dropout_prob,
                # Make controller plane-aware to avoid over-provisioning
                max_planes=self.num_planes,
            )
            # Inform controller about number of planes
            self.controller.num_planes = self.num_planes
            # Controlled Cellular Policy (bounded residual application of controller outputs)
            self.ccp = CCP(
                C=num_units,
                gamma_max=0.25,
                beta_max=0.5,
                energy_weight=0.0,
                log_every=0,
            )

    def set_task_id(self, task_id: str):
        if self.task_bank is not None:
            self.task_bank.ensure(task_id)
        self.active_task_id = task_id

    def hard_sigmoid(self, x_in):
        self.add_saturation_cost(x_in)
        x = x_in * 0.5 + 0.5
        return torch.clamp(x, 0.0, 1.0)

    def hard_tanh(self, x):
        self.add_saturation_cost(x)
        return torch.clamp(x, -1.0, 1.0)

    def add_saturation_cost(self, var):
        sat_loss = F.relu(torch.abs(var) - self.saturation_limit)
        cost = torch.sum(sat_loss)
        self.saturation_costs.append(cost)

    @staticmethod
    def _one_hot_2d(x_idx, n_vocab):
        # x_idx: (B,H,W) -> (B,n_vocab,H,W)
        oh = F.one_hot(x_idx.long(), num_classes=n_vocab).float()  # (B,H,W,n_vocab)
        # Treat index 0 as padding -> contribute no signal
        oh[..., 0] = 0.0
        return oh.permute(0,3,1,2)

    def forward(self, x_in_indices, steps, support: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        """
        x_in_indices: (B, H, W) integer tokens, **0 is padding** (we'll shift real ARC colors by +1).
        steps: number of recurrent updates.
        support: Optional K-shot support list [(x_in, x_out), ...] with tensors (B,H,W) each.
        Returns:
          logits: (B, H, W, n_classes)
          sat_loss: scalar
        """
        self.saturation_costs = []

        B, H, W = x_in_indices.shape

        x_onehot = self._one_hot_2d(x_in_indices, self.n_input)         # (B,Cin,H,W)
        cur = self.input_layer(x_onehot)                                 # (B,C,H,W)
        cur = self.hard_tanh(cur)                                        # (B,C,H,W)

        # 2D prompt: add program-conditioned plane to input features
        if self.conditioning == "prompt2d":
            assert support is not None and len(support) > 0, "prompt2d conditioning requires support K-shot pairs"
            prompt_bchw = self.prompt2d(support, H=H, W=W)               # (B,C,H,W)
            cur = cur + prompt_bchw
            cur = self.hard_tanh(cur)

        cur = cur.permute(0,2,3,1)                                       # (B,H,W,C)
        input_anchor = cur  # preserve BHWC view of input features

        # Initialize multi-plane state (B,P,H,W,C)
        if self.num_planes > 1:
            zeros = torch.zeros_like(cur).unsqueeze(1).repeat(1, self.num_planes - 1, 1, 1, 1)
            cur = torch.cat([cur.unsqueeze(1), zeros], dim=1)
        else:
            cur = cur.unsqueeze(1)

        if self.training and self.dropout_prob > 0:
            cur = F.dropout(cur, p=self.dropout_prob)
            input_anchor = F.dropout(input_anchor, p=self.dropout_prob)

        # Precompute FiLM parameters z -> gamma/beta (applied each step)
        gamma = beta = None
        if self.conditioning == "film":
            assert support is not None and len(support) > 0, "FiLM conditioning requires support K-shot pairs"
            z = self.program_encoder(support)                              # (B,C)
            gamma = self.film_gamma(z)[:, None, None, None, :]            # (B,1,1,1,C)
            beta  = self.film_beta(z)[:, None, None, None, :]             # (B,1,1,1,C)
        elif (self.task_bank is not None) and self.task_latent_use_film and (self.active_task_id is not None):
            zt = self.task_bank.get(self.active_task_id, device=cur.device, dtype=cur.dtype)
            zt_b = zt[None, :].expand(B, -1)
            gτ, bτ = self.task_bank.film(zt_b)
            gamma, beta = gτ, bτ


        # Controller inputs: supports (legacy) OR task-latent τ (lightweight)
        support_tokens = None
        task_vec = None
        task_prefix = None
        if self.conditioning == "controller":
            if (self.task_bank is not None) and (self.active_task_id is not None):
                z = self.task_bank.get(self.active_task_id, device=cur.device, dtype=cur.dtype)  # [D]
                z_b = z[None, :].expand(B, -1)  # [B,D]
                task_vec = self.task_bank.ctrl_steer(z_b)                 # [B,C]
                task_prefix = self.task_bank.ctrl_prefix(z_b)             # [B,Gτ,C] or None
            else:
                # fallback to legacy support encoding if τ is absent
                if support is not None and len(support) > 0:
                    K, s_in, s_out = self._parse_support(support)
                    toks = []
                    for k in range(K):
                        tin = self.support_to_latents._encode(s_in[k])   # [B,S,C]
                        tout = self.support_to_latents._encode(s_out[k]) # [B,S,C]
                        toks.append(tin + tout)
                    support_tokens = torch.cat(toks, dim=1) if len(toks) > 0 else None
                else:
                    support_tokens = None

        # Optional controller scratch state
        ctrl_state = None

        ctrl_cache = None
        ccp_energy_total = 0.0
        for t in range(steps):
            # Enforce input anchor plane before controller reads
            if self.force_input_plane:
                pidx = min(self.input_plane_index, self.num_planes - 1)
                cur[:, pidx, ...] = input_anchor
            # Controller: compute control heads and apply via CCP before the cell update
            if self.conditioning == "controller":
                if (t % self.controller_update_every == 0) or (ctrl_cache is None):
                    ctrl_out, ctrl_state = self.controller(cur, support_tokens, step_t=t,
                                                           scratchpad=ctrl_state,
                                                           task_vec=task_vec, task_prefix=task_prefix)
                    ctrl_cache = ctrl_out
                else:
                    ctrl_out = ctrl_cache

                # Prune heads according to enabled flags (only FiLM remains)
                if not self.controller_enable_film:
                    ctrl_out.pop('gamma_p', None); ctrl_out.pop('beta_p', None)

                cur, ccp_energy, _stats = self.ccp.apply_multi(cur, ctrl_out, step_t=t)
                if isinstance(ccp_energy, torch.Tensor):
                    ccp_energy_total = ccp_energy_total + ccp_energy

            # Apply FiLM after each update (token-wise)
            if gamma is not None and beta is not None:
                cur = cur * (1.0 + gamma) + beta
                # Thin saturation loss collection if requested
                if self.sat_every is None or (t % max(self.sat_every, 1) == 0):
                    cur = self.hard_tanh(cur)
                else:
                    cur = torch.clamp(cur, -1.0, 1.0)

            # Step the recurrent cell once per loop (vectorized over planes)
            step_fn = getattr(self, "_compiled_step", None) or (lambda x: self._step_planes(x))
            cur = step_fn(cur)

            # Optional 2D-aware global mixing every k steps
            if (self.mixer is not None) and ((t % self.mixer_every) == 0):
                cur = self.mixer(cur)

            # Re-enforce input anchor after updates to keep it constant
            if self.force_input_plane:
                pidx = min(self.input_plane_index, self.num_planes - 1)
                cur[:, pidx, ...] = input_anchor

            # No per-step gate bias to clear

            # Optional truncation inside forward (for non-TBPTT training)
            if self.detach_every and self.training and ((t + 1) % self.detach_every == 0) and (t + 1 < steps):
                cur = cur.detach()

        # Compose planes for head
        if self.conditioning == "controller" and ctrl_cache is not None and ('alpha' in ctrl_cache):
            alpha = ctrl_cache['alpha'][:, :, None, None, None]  # [B,P,1,1,1]
            cur_read = (alpha * cur).sum(dim=1)                  # [B,H,W,C]
        else:
            cur_read = cur[:, 0]
        logits = self.output_layer(cur_read.permute(0,3,1,2))
        logits = logits.permute(0,2,3,1)

        total_elems = B * self.num_planes * H * W * max(steps, 1)
        if len(self.saturation_costs) > 0:
            sat_sum = torch.sum(torch.stack(self.saturation_costs))
        else:
            sat_sum = torch.tensor(0.0, device=cur.device, dtype=cur.dtype)
        sat_loss = sat_sum / max(total_elems, 1)
        # Include CCP energy regularization if used
        if self.conditioning == "controller" and isinstance(ccp_energy_total, torch.Tensor):
            sat_loss = sat_loss + ccp_energy_total
        return logits, sat_loss

    def _step_planes(self, cur_bphwc: torch.Tensor) -> torch.Tensor:
        """Vectorize DCGRU2D across planes by fusing planes into batch."""
        B, P, H, W, C = cur_bphwc.shape
        x = cur_bphwc.reshape(B * P, H, W, C)
        y = self.cell(x)
        return y.reshape(B, P, H, W, C)

    def forward_chunk(self,
                      x_in_indices: torch.Tensor,
                      steps: int,
                      carry_state: Optional[torch.Tensor] = None,
                      support: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                      return_state: bool = False):
        """
        Chunked forward that supports Truncated BPTT.
          - If carry_state is None: initializes state from x_in_indices (like forward()).
          - Otherwise: starts from BHWC/BPHWC `carry_state` and advances `steps` updates.
        Returns (logits, sat_loss[, state]).
        """
        self.saturation_costs = []
        B, H, W = x_in_indices.shape

        if carry_state is None:
            x_onehot = self._one_hot_2d(x_in_indices, self.n_input)         # (B,Cin,H,W)
            cur = self.input_layer(x_onehot)                                 # (B,C,H,W)
            # Record input saturation once per chunk
            cur = self.hard_tanh(cur)

            if self.conditioning == "prompt2d":
                assert support is not None and len(support) > 0, "prompt2d conditioning requires support K-shot pairs"
                prompt_bchw = self.prompt2d(support, H=H, W=W)               # (B,C,H,W)
                cur = self.hard_tanh(cur + prompt_bchw)

            cur = cur.permute(0,2,3,1)                                       # (B,H,W,C)
            input_anchor = cur
            # Initialize multi-plane state
            if self.num_planes > 1:
                zeros = torch.zeros_like(cur).unsqueeze(1).repeat(1, self.num_planes - 1, 1, 1, 1)
                cur = torch.cat([cur.unsqueeze(1), zeros], dim=1)            # (B,P,H,W,C)
            else:
                cur = cur.unsqueeze(1)
            if self.training and self.dropout_prob > 0:
                cur = F.dropout(cur, p=self.dropout_prob)
                input_anchor = F.dropout(input_anchor, p=self.dropout_prob)
        else:
            cur = carry_state  # expected (B,P,H,W,C) or (B,H,W,C)
            if cur.dim() == 4:
                cur = cur.unsqueeze(1)
            # Recompute input anchor from current inputs for this chunk
            x_onehot = self._one_hot_2d(x_in_indices, self.n_input)
            anc = self.input_layer(x_onehot)
            anc = self.hard_tanh(anc)
            input_anchor = anc.permute(0,2,3,1)
            if self.training and self.dropout_prob > 0:
                input_anchor = F.dropout(input_anchor, p=self.dropout_prob)

        # Precompute FiLM gamma/beta if used
        gamma = beta = None
        if self.conditioning == "film":
            assert support is not None and len(support) > 0, "FiLM conditioning requires support K-shot pairs"
            z = self.program_encoder(support)                              # (B,C)
            gamma = self.film_gamma(z)[:, None, None, None, :]            # (B,1,1,1,C)
            beta  = self.film_beta(z)[:, None, None, None, :]             # (B,1,1,1,C)
        elif (self.task_bank is not None) and self.task_latent_use_film and (self.active_task_id is not None):
            zt = self.task_bank.get(self.active_task_id, device=input_anchor.device, dtype=input_anchor.dtype)
            zt_b = zt[None, :].expand(B, -1)
            gτ, bτ = self.task_bank.film(zt_b)
            gamma, beta = gτ, bτ


        # Controller inputs: supports (legacy) OR task-latent τ (lightweight)
        support_tokens = None
        task_vec = None
        task_prefix = None
        if self.conditioning == "controller":
            if (self.task_bank is not None) and (self.active_task_id is not None):
                z = self.task_bank.get(self.active_task_id, device=cur.device, dtype=cur.dtype)  # [D]
                z_b = z[None, :].expand(B, -1)
                task_vec = self.task_bank.ctrl_steer(z_b)
                task_prefix = self.task_bank.ctrl_prefix(z_b)
            else:
                if support is not None and len(support) > 0:
                    K, s_in, s_out = self._parse_support(support)
                    toks = []
                    for k in range(K):
                        tin = self.support_to_latents._encode(s_in[k])
                        tout = self.support_to_latents._encode(s_out[k])
                        toks.append(tin + tout)
                    support_tokens = torch.cat(toks, dim=1) if len(toks) > 0 else None
                else:
                    support_tokens = None

        # Advance steps
        ctrl_cache = None
        ccp_energy_total = 0.0
        for t in range(steps):
            if self.force_input_plane:
                pidx = min(self.input_plane_index, self.num_planes - 1)
                cur[:, pidx, ...] = input_anchor
            if self.conditioning == "controller":
                if (t % self.controller_update_every == 0) or (ctrl_cache is None):
                    ctrl_out, _ = self.controller(cur, support_tokens, step_t=t,
                                                  scratchpad=None,
                                                  task_vec=task_vec, task_prefix=task_prefix)
                    ctrl_cache = ctrl_out
                else:
                    ctrl_out = ctrl_cache

                # Prune heads according to enabled flags
                if not self.controller_enable_film:
                    ctrl_out.pop('gamma_p', None); ctrl_out.pop('beta_p', None)
                cur, ccp_energy, _stats = self.ccp.apply_multi(cur, ctrl_out, step_t=t)
                if isinstance(ccp_energy, torch.Tensor):
                    ccp_energy_total = ccp_energy_total + ccp_energy
            # Only DCGRU2D supported: step the cell (optionally compiled)
            step_fn = getattr(self, "_compiled_step", None) or (lambda x: self._step_planes(x))

            if self.use_checkpoint and self.training and cur.requires_grad:
                cur = _ckpt(step_fn, cur, use_reentrant=False)
            else:
                cur = step_fn(cur)

            # Optional 2D-aware global mixing every k steps
            if (self.mixer is not None) and ((t % self.mixer_every) == 0):
                cur = self.mixer(cur)

            # No per-step gate bias to clear

            if gamma is not None and beta is not None:
                cur = cur * (1.0 + gamma) + beta
                if self.sat_every is None or (t % max(self.sat_every, 1) == 0):
                    cur = self.hard_tanh(cur)
                else:
                    cur = torch.clamp(cur, -1.0, 1.0)

            if self.force_input_plane:
                pidx = min(self.input_plane_index, self.num_planes - 1)
                cur[:, pidx, ...] = input_anchor

        # Head and losses for this chunk
        if self.conditioning == "controller" and ctrl_cache is not None and ('alpha' in ctrl_cache):
            alpha = ctrl_cache['alpha'][:, :, None, None, None]
            cur_read = (alpha * cur).sum(dim=1)
        else:
            cur_read = cur[:, 0]
        logits = self.output_layer(cur_read.permute(0,3,1,2)).permute(0,2,3,1)
        total_elems = B * self.num_planes * H * W * max(steps, 1)
        if len(self.saturation_costs) > 0:
            sat_sum = torch.sum(torch.stack(self.saturation_costs))
        else:
            sat_sum = torch.tensor(0.0, device=cur.device, dtype=cur.dtype)
        sat_loss = sat_sum / max(total_elems, 1)
        if self.conditioning == "controller" and isinstance(ccp_energy_total, torch.Tensor):
            sat_loss = sat_loss + ccp_energy_total
        return (logits, sat_loss, cur) if return_state else (logits, sat_loss)

    @staticmethod
    def _parse_support(support: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Normalize support input format.
        Returns K, support_in [K,B,H,W], support_out [K,B,H,W].
        Accepts list of (inp, out) pairs or a tuple (support_ins, support_outs).
        """
        if support is None:
            return 0, None, None
        # Already in tensors (K,B,H,W)
        if isinstance(support, tuple) and len(support) == 2 and isinstance(support[0], torch.Tensor):
            s_in, s_out = support
            assert s_in.shape == s_out.shape
            K = s_in.shape[0]
            return K, s_in, s_out
        # List of pairs
        if isinstance(support, (list, tuple)):
            Ks = len(support)
            s_in = torch.stack([pair[0] for pair in support], dim=0)
            s_out = torch.stack([pair[1] for pair in support], dim=0)
            return Ks, s_in, s_out
        raise ValueError("Unsupported support format; expected list[(in,out)] or (support_in, support_out)")


class DCGRU2D(nn.Module):
    def __init__(self, num_units, kernel_size=3, dropout=0.0, neighborhood="von_neumann", toroidal=False):
        super().__init__()
        self.num_units = num_units
        self.kernel_size = kernel_size
        self.dropout = dropout
        assert kernel_size % 2 == 1, "Use odd kernel_size (e.g., 3)."
        assert neighborhood in ("von_neumann", "moore")
        self.neighborhood = neighborhood
        self.toroidal = toroidal

        pad = kernel_size // 2
        pad_mode = "circular" if self.toroidal else "zeros"
        self.conv_r = nn.Conv2d(num_units, num_units, kernel_size, padding=pad, padding_mode=pad_mode)
        self.conv_c = nn.Conv2d(num_units, num_units, kernel_size, padding=pad, padding_mode=pad_mode)
        self.conv_g = nn.Conv2d(num_units, num_units, kernel_size, padding=pad, padding_mode=pad_mode)

        shift = self._create_shift_filter(num_units, neighborhood)
        self.register_buffer("shift_filter", shift)  # (C,1,3,3), moves with .to(device)
        # No per-step external gate biases in simplified setup

    @staticmethod
    def _direction_kernel(pos):
        k = torch.zeros(3,3, dtype=torch.float32)
        k[pos] = 1.0
        return k

    def _create_shift_filter(self, C, neighborhood):
        dirs_vn = [("stay",(1,1)), ("up",(0,1)), ("down",(2,1)),
                   ("left",(1,0)), ("right",(1,2))]
        dirs_moore = dirs_vn + [("ul",(0,0)), ("ur",(0,2)), ("dl",(2,0)), ("dr",(2,2))]
        dirs = dirs_vn if neighborhood == "von_neumann" else dirs_moore
        D = len(dirs)
        counts = [C // D + (1 if i < (C % D) else 0) for i in range(D)]
        kernels = []
        for i, (_, rc) in enumerate(dirs):
            k = self._direction_kernel(rc).unsqueeze(0).unsqueeze(0)     # (1,1,3,3)
            kernels.append(k.repeat(counts[i], 1, 1, 1))                  # (#ch_i,1,3,3)
        weight = torch.cat(kernels, dim=0)                                # (C,1,3,3)
        return weight

    @staticmethod
    def hard_sigmoid(x):  # CellGRU collects saturation, no tracking here
        return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)

    @staticmethod
    def hard_tanh(x):
        return torch.clamp(x, -1.0, 1.0)

    def forward(self, mem_bhwc):
        B,H,W,C = mem_bhwc.shape
        mem = mem_bhwc.permute(0,3,1,2)                                   # (B,C,H,W)
 
        # --- shift with optional wrap-around (toroidal) ---
        if self.toroidal:
            # circular pad by 1 on all sides, then valid conv
            mem_padded = F.pad(mem, (1,1,1,1), mode="circular")         # (B,C,H+2,W+2)
            mem_shifted = F.conv2d(mem_padded, self.shift_filter, padding=0, groups=self.num_units)
        else:
            # regular zero-padding shift
            mem_shifted = F.conv2d(mem, self.shift_filter, padding=1, groups=self.num_units)
 
        # Gate-bias removed (no external per-step biases)
        r_bias = z_bias = None

        reset = self.conv_r(mem)
        if r_bias is not None and r_bias.shape[0] == reset.shape[0]:
            reset = reset + r_bias
        reset = self.hard_sigmoid(reset + 0.5)
        cand  = self.hard_tanh(self.conv_c(reset * mem))

        if self.training and self.dropout > 0:
            cand = F.dropout(cand, p=self.dropout)

        gate_pre = self.conv_g(mem)
        # z_bias unused (removed)
        gate  = self.hard_sigmoid(gate_pre + 0.7)
        out   = gate * mem_shifted + (1.0 - gate) * cand                  # (B,C,H,W)
        return out.permute(0,2,3,1)                                       # (B,H,W,C)


class GlobalAttnGRU2D(nn.Module):
    """
    Drop-in alternative to DCGRU2D:
      - Flattens HxW -> sequence
      - Adds 2D sinusoidal positions
      - Applies multi-head self-attention
      - Uses GRU-style gated write to update the memory
    Complexity: O((H*W)^2).
    """
    def __init__(self, num_units, num_heads=8, dropout=0.0):
        super().__init__()
        self.c = num_units
        self.mha = MHA_SDPA(c=num_units, num_heads=num_heads, dropout=dropout)
        self.ln_in  = nn.LayerNorm(num_units)
        self.ln_out = nn.LayerNorm(num_units)

        # GRU-style gates working tokenwise: z (update), r (reset)
        self.gates = nn.Linear(2 * num_units, 2 * num_units)  # [x, attn] -> [z,r]
        self.cand  = nn.Linear(2 * num_units, num_units)      # [r*x, attn] -> h_tilde
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _pos_1d(L, dim):
        """Standard sin-cos; dim must be even."""
        assert dim % 2 == 0
        pos = torch.arange(L, dtype=torch.float32).unsqueeze(1)  # [L,1]
        i   = torch.arange(dim // 2, dtype=torch.float32).unsqueeze(0)  # [1,d/2]
        denom = torch.exp(-math.log(10000.0) * (2 * i) / dim)
        ang = pos * denom  # [L, d/2]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # [L, dim]

    @staticmethod
    def _pos_2d(H, W, dim):
        """Concat row+col encodings. dim must be divisible by 2."""
        assert dim % 2 == 0
        dim_h = dim // 2
        dim_w = dim - dim_h
        rh = GlobalAttnGRU2D._pos_1d(H, dim_h)                     # [H, dim_h]
        cw = GlobalAttnGRU2D._pos_1d(W, dim_w)                     # [W, dim_w]
        pos = (rh[:, None, :].repeat(1, W, 1),  # [H,W,dim_h]
               cw[None, :, :].repeat(H, 1, 1))  # [H,W,dim_w]
        pos = torch.cat(pos, dim=2).view(H*W, dim)                 # [HW, dim]
        return pos

    @staticmethod
    def _hard_sigmoid(x):
        return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)

    def forward(self, mem_bhwc):
        B, H, W, C = mem_bhwc.shape
        x = mem_bhwc.view(B, H*W, C)                               # [B,S,C], S=H*W
        pos = Pos2DCache.get(H, W, C, x.device, x.dtype)           # [S,C]

        x_in   = self.ln_in(x + pos[None, :, :])                   # [B,S,C]
        attn   = self.mha(x_in, x_in)                              # [B,S,C]
        attn   = self.dropout(attn)
        y      = self.ln_out(attn)                                 # [B,S,C]

        # GRU-style gated write (tokenwise)
        x_flat = x_in.reshape(B*H*W, C)                            # [B*S,C]
        y_flat = y.reshape(B*H*W, C)                               # [B*S,C]

        z_r = self.gates(torch.cat([x_flat, y_flat], dim=-1))      # [B*S,2C]
        z, r = torch.chunk(z_r, 2, dim=-1)
        z = self._hard_sigmoid(z)
        r = self._hard_sigmoid(r)

        h_tilde = torch.tanh(self.cand(torch.cat([r * x_flat, y_flat], dim=-1)))
        out_flat = (1.0 - z) * x_flat + z * h_tilde                # [B*S,C]
        out = out_flat.view(B, H, W, C)
        return out

class ProgramEncoder(nn.Module):
    """
    Encode K support (input, output) grid pairs into a task embedding z in R^C.
    Uses the model's input_layer for shared embedding, then pools and projects.
    """
    def __init__(self, input_layer: nn.Conv2d, c: int, hidden: int = 256):
        super().__init__()
        self.embed = input_layer  # shared 1x1 conv to C
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(2 * c, hidden), nn.ReLU(),
            nn.Linear(hidden, c)
        )

    def _encode(self, x_idx: torch.Tensor) -> torch.Tensor:
        # x_idx: [B,H,W] -> features [B,C]
        B, H, W = x_idx.shape
        oh = F.one_hot(x_idx.long(), num_classes=self.embed.in_channels).float()
        oh[..., 0] = 0.0  # zero pad-channel
        x_oh = oh.permute(0,3,1,2)
        feats = torch.tanh(self.embed(x_oh))  # [B,C,H,W]
        pooled = self.pool(feats).squeeze(-1).squeeze(-1)  # [B,C]
        return pooled

    def forward(self, support: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        K = len(support)
        fi_list = []
        fo_list = []
        for k in range(K):
            s_in, s_out = support[k]
            fi = self._encode(s_in)
            fo = self._encode(s_out)
            fi_list.append(fi); fo_list.append(fo)
        fi = torch.stack(fi_list, dim=0)   # [K,B,C]
        fo = torch.stack(fo_list, dim=0)   # [K,B,C]
        B = fi.shape[1]
        feats = torch.cat([fi, fo], dim=-1).mean(dim=0)  # [B, 2C]
        z = self.proj(feats)  # [B,C]
        return z


class Prompt2D(nn.Module):
    """
    Learn a global prompt plane and modulate it by a program embedding.
    Returns a spatial prompt to add to input features (B,C,H,W).
    """
    def __init__(self, C: int, Hmax: int = 18, Wmax: int = 18, program_encoder: Optional[ProgramEncoder] = None):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(1, C, Hmax, Wmax))
        nn.init.xavier_uniform_(self.prompt)
        self.gen = program_encoder

    def forward(self, support: List[Tuple[torch.Tensor, torch.Tensor]], H: int, W: int) -> torch.Tensor:
        assert self.gen is not None, "Prompt2D requires a ProgramEncoder"
        z = self.gen(support)  # [B,C]
        scale = torch.tanh(z)[:, :, None, None]  # [B,C,1,1]
        p = self.prompt
        if p.shape[-2:] != (H, W):
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
        return scale * p  # [B,C,H,W]


class SupportLatentGenerator(nn.Module):
    """
    Convert K-shot support (input, output) grids into per-batch latent slots [B,K_lat,C].
    Uses a single cross-attention step from learnable latent queries to support tokens.
    """
    def __init__(self, input_layer: nn.Conv2d, C: int, num_latents: int = 64, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed = input_layer  # shared embedding to C
        self.C = C
        self.num_latents = num_latents
        self.latent_queries = nn.Parameter(torch.randn(num_latents, C) / math.sqrt(C))
        self.ln_tok = nn.LayerNorm(C)
        self.ln_lat = nn.LayerNorm(C)
        self.cross = MHA_SDPA(C, num_heads, dropout=dropout)

    @staticmethod
    def _pos_2d(H, W, dim):
        return GlobalAttnGRU2D._pos_2d(H, W, dim)

    def _encode(self, x_idx: torch.Tensor) -> torch.Tensor:
        # [B,H,W] -> [B,H*W,C]
        B, H, W = x_idx.shape
        oh = F.one_hot(x_idx.long(), num_classes=self.embed.in_channels).float()
        oh[..., 0] = 0.0  # zero pad-channel
        x_oh = oh.permute(0,3,1,2)
        feats = torch.tanh(self.embed(x_oh))  # [B,C,H,W]
        seq = feats.permute(0,2,3,1).reshape(B, H*W, self.C)
        pos = Pos2DCache.get(H, W, self.C, seq.device, seq.dtype)[None, :, :]  # [1,S,C]
        return self.ln_tok(seq + pos)  # [B,S,C]

    def forward(self, support: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        K = len(support)
        # Build token sequence from K pairs by encoding input and output then summing
        toks = []
        for k in range(K):
            s_in, s_out = support[k]
            tin = self._encode(s_in)   # [B,S,C]
            tout = self._encode(s_out) # [B,S,C]
            toks.append(tin + tout)
        tokens = torch.cat(toks, dim=1)  # [B, K*S, C]

        # Latent queries per batch
        B = tokens.shape[0]
        lat = self.latent_queries[None, :, :].expand(B, -1, -1)  # [B,K_lat,C]

        lat_q = self.ln_lat(lat)  # [B,K_lat,C]
        lat_out = self.cross(lat_q, tokens)  # [B,K_lat,C]
        return lat_out



class GlobalMixer2D(nn.Module):
    """Full self-attention over P*H*W tokens; 2D pos + plane tags; gated residual."""
    def __init__(self, C: int, heads: int = 8, depth: int = 1, dropout: float = 0.0):
        super().__init__()
        self.C = C
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln": nn.LayerNorm(C),
                "mha": MHA_SDPA(C, heads, dropout),
                "ff": nn.Sequential(
                    nn.LayerNorm(C),
                    nn.Linear(C, 4*C), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(4*C, C)
                )
            }) for _ in range(max(1, depth))
        ])
        self.res_scale = nn.Parameter(torch.tensor(0.2))
        self.plane_emb = nn.Embedding(256, C)

    def forward(self, cur_bphwc: torch.Tensor) -> torch.Tensor:
        B, P, H, W, C = cur_bphwc.shape
        seq = cur_bphwc.view(B, P*H*W, C)
        pos = Pos2DCache.get(H, W, C, seq.device, seq.dtype)[None, :, :].repeat(1, P, 1)
        ptag = self.plane_emb(torch.arange(P, device=seq.device))[:, None, :].repeat(1, H*W, 1)
        ptag = ptag.reshape(P*H*W, C)[None, :, :]
        h = seq + pos + ptag
        for blk in self.blocks:
            h = h + blk["mha"](blk["ln"](h), blk["ln"](h))
            h = h + blk["ff"](h)
        out = h.view(B, P, H, W, C)
        return cur_bphwc + torch.tanh(self.res_scale) * out


## AxialMixer2D, CrossPoolMixer2D, and FullPooledMixer2D removed: deprecated or underperforming.
