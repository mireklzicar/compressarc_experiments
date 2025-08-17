import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSoftmaxBottleneck(nn.Module):
    """
    Token-wise categorical bottleneck for ViT-like autoencoders.
    Input:  (B, N, D) token embeddings
    Output: (B, N, D) quantized embeddings via learned codebook and Gumbel-Softmax.

    Losses returned in aux:
      - kl_uniform: encourage usage of all codes (avoid collapse)
      - entropy: per-token categorical entropy (optionally penalize to harden selections)
      - perplexity: codebook usage metric
    """
    def __init__(
        self,
        embed_dim: int,
        codebook_size: int = 512,
        temp_init: float = 1.0,
        temp_min: float = 0.2,
        anneal_rate: float = 5e-5,
        proj_bias: bool = True,
        use_straight_through: bool = True,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = codebook_size
        self.use_st = use_straight_through

        # Encoder-to-logits over codes
        self.to_logits = nn.Linear(embed_dim, codebook_size, bias=proj_bias)

        # Codebook: (K, D)
        self.codebook = nn.Parameter(torch.randn(codebook_size, embed_dim) * init_scale)

        # Temperature schedule
        self.register_buffer("temp", torch.tensor(temp_init))
        self.temp_init = temp_init
        self.temp_min = temp_min
        self.anneal_rate = anneal_rate

        # Track steps for annealing (buffer so it moves with the model)
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def step_anneal(self):
        # Exponential anneal until temp_min
        if self.temp > self.temp_min:
            new_t = self.temp_init * math.exp(-self.anneal_rate * int(self.steps.item()))
            self.temp.copy_(torch.tensor(max(self.temp_min, new_t)))

    def forward(self, x, temp: float = None, hard: bool = None):
        """
        x: (B, N, D)
        temp: override temperature
        hard: override hard selection (straight-through). If None, use self.use_st.
        """
        if self.training:
            # Step and anneal temperature
            self.steps += 1
            self.step_anneal()

        tau = self.temp.item() if temp is None else temp
        hard = self.use_st if hard is None else hard

        B, N, D = x.shape
        logits = self.to_logits(x)             # (B, N, K)
        # Soft categorical with Gumbel noise
        probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)  # (B, N, K)

        if hard:
            # Straight-through: hard one-hot in forward, soft in backward
            idx = probs.argmax(dim=-1)                             # (B, N)
            onehot = F.one_hot(idx, num_classes=self.K).type_as(probs)
            y = onehot + (probs - probs.detach())                  # (B, N, K)
        else:
            idx = logits.argmax(dim=-1)  # for monitoring
            y = probs

        # Quantize: convex combo of codebook rows
        # (B, N, K) @ (K, D) -> (B, N, D)
        z = torch.einsum('bnk,kd->bnd', y, self.codebook)

        # Auxiliary stats / losses
        eps = 1e-9
        # Encourage usage across codes to be close to uniform
        avg_probs = probs.mean(dim=(0, 1))                         # (K,)
        kl_uniform = (avg_probs * (avg_probs.add(eps).log() - math.log(1.0 / self.K))).sum()

        # Per-token entropy: can penalize to encourage “hard” choices
        entropy = -(probs * probs.add(eps).log()).sum(dim=-1).mean()

        # Perplexity for monitoring (higher is better up to K)
        perplexity = torch.exp(-(avg_probs.add(eps) * avg_probs.add(eps).log()).sum())

        aux = {
            "indices": idx,                  # (B, N) int64
            "probs": probs,                  # (B, N, K)
            "kl_uniform": kl_uniform,        # scalar
            "entropy": entropy,              # scalar
            "perplexity": perplexity,        # scalar
            "temperature": torch.tensor(tau, device=x.device),
        }
        return z, aux