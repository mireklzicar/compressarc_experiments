# src/models/ccp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CCP(nn.Module):
    """
    Controlled Cellular Policy: safely apply bounded control fields to BHWC state.
    Supports FiLM (gamma,beta) only. Applies residual updates and returns an
    energy penalty for regularization.
    """
    def __init__(self,
                 C: int,
                 gamma_max: float = 0.25,
                 beta_max: float = 0.5,
                 energy_weight: float = 1e-4,
                 log_every: int = 0
                 ):
        super().__init__()
        self.C = C
        self.gamma_max = float(gamma_max)
        self.beta_max = float(beta_max)
        self.energy_weight = float(energy_weight)
        self.log_every = int(max(0, log_every))

    @staticmethod
    def _bound(x, m):
        return torch.tanh(x) * m

    def apply(self, cur_bhwc: torch.Tensor, ctrl_out: dict, H: int, W: int, cell=None, step_t: int = 0):
        """
        cur_bhwc: [B,H,W,C]
        ctrl_out: dict from TransformerController2D (gamma,beta)
        cell: ignored
        Returns: (cur_bhwc, energy_loss, stats or None)
        """
        B, Hc, Wc, C = cur_bhwc.shape
        assert Hc == H and Wc == W and C == self.C

        # Heads enabled are determined upstream (by controller_enable_* flags).
        # We bound their values and compute energy if present in ctrl_out.
        energy_terms = []
        stats = {}

        # FiLM head
        if ('gamma' in ctrl_out) and ('beta' in ctrl_out):
            gamma = self._bound(ctrl_out['gamma'], self.gamma_max)   # [B,1,1,C]
            beta  = self._bound(ctrl_out['beta'],  self.beta_max)    # [B,1,1,C]
            cur_bhwc = cur_bhwc + (gamma * cur_bhwc + beta)
            energy_terms += [gamma.square().mean(), beta.square().mean()]
            if self.log_every and (step_t % self.log_every == 0):
                stats['gamma_rms'] = gamma.float().pow(2).mean().sqrt().item()
                stats['beta_rms']  = beta.float().pow(2).mean().sqrt().item()

        # Energy
        energy_loss = 0.0 * cur_bhwc.sum()
        if len(energy_terms) > 0 and self.energy_weight > 0.0:
            energy_loss = self.energy_weight * sum(energy_terms)

        return cur_bhwc, energy_loss, stats if stats else None

    def apply_multi(self, cur_bphwc: torch.Tensor, ctrl_out: dict, step_t: int = 0):
        """
        Multi-plane variant.
        cur_bphwc: [B,P,H,W,C]
        ctrl_out: may include 'gamma_p' [B,P,1,1,C], 'beta_p' [B,P,1,1,C], optional 'router' [B,P,P]
        Returns: (cur_bphwc, energy_loss, stats or None)
        """
        B, P, H, W, C = cur_bphwc.shape
        energy_terms = []
        stats = {}

        if ('gamma_p' in ctrl_out) and ('beta_p' in ctrl_out):
            gamma = self._bound(ctrl_out['gamma_p'], self.gamma_max)  # [B,P,1,1,C]
            beta  = self._bound(ctrl_out['beta_p'],  self.beta_max)   # [B,P,1,1,C]
            cur_bphwc = cur_bphwc + gamma * cur_bphwc + beta
            energy_terms += [gamma.square().mean(), beta.square().mean()]
            if self.log_every and (step_t % self.log_every == 0):
                stats['gamma_rms'] = gamma.float().pow(2).mean().sqrt().item()
                stats['beta_rms']  = beta.float().pow(2).mean().sqrt().item()

        if 'router' in ctrl_out:
            # new_planes[p] = sum_q R[p,q] * cur[q]
            R = ctrl_out['router']  # [B,P,P]
            cur_flat = cur_bphwc.view(B, P, H * W * C)
            mixed = torch.matmul(R, cur_flat)  # [B,P,H*W*C]
            cur_bphwc = mixed.view(B, P, H, W, C)

        energy_loss = 0.0 * cur_bphwc.sum()
        if len(energy_terms) > 0 and self.energy_weight > 0.0:
            energy_loss = self.energy_weight * sum(energy_terms)
        return cur_bphwc, energy_loss, stats if stats else None
