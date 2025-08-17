import math
import torch
import torch.nn as nn

def reconstruction_loss(pred, target, kind="l2"):
    if kind == "l1":
        return (pred - target).abs().mean()
    if kind == "l2":
        return ((pred - target) ** 2).mean()
    if kind == "cross_entropy":
        # pred: (B, N, vocab_size), target: (B, N)
        import torch.nn.functional as F
        return F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1))
    raise ValueError(kind)

def compute_loss(out, target, aux, w_kl=1e-2, w_entropy=1e-3, w_size=1.0, maximize_code_usage=True):
    """
    - w_kl encourages global uniform codebook usage (prevents collapse).
    - w_entropy penalizes per-token entropy to harden assignments => compression.
      If you want softer assignments early, linearly ramp this weight from 0.
    - w_size penalizes the model for predicting a different number of tokens than the target.
    """
    rec = reconstruction_loss(out, target, kind="cross_entropy")
    klu = aux["kl_uniform"]
    ent = aux["entropy"]

    # Penalize size mismatch
    pred_len = torch.tensor(out.shape[1], device=out.device, dtype=torch.float32)
    target_len = torch.tensor(target.shape[1], device=target.device, dtype=torch.float32)
    size_loss = ((pred_len - target_len) ** 2)

    # Encourage uniform usage => minimize KL(avg_probs || uniform)
    loss = rec + w_kl * klu + w_size * size_loss

    # Encourage low per-token entropy (hard choices) => minimize entropy
    # If you prefer to keep things soft early on, ramp w_entropy from 0 to final value.
    loss = loss + w_entropy * ent

    # Optional: report bits-per-token (approx) as monitoring
    # bpt ~ H(avg_probs) / ln(2), where H is entropy of code usage distribution
    with torch.no_grad():
        avg = aux["probs"].mean(dim=(0, 1))
        bpt = -(avg * (avg + 1e-9).log()).sum() / math.log(2.0)
    metrics = {
        "loss_total": loss.detach(),
        "loss_rec": rec.detach(),
        "loss_size": size_loss.detach(),
        "kl_uniform": klu.detach(),
        "entropy": ent.detach(),
        "perplexity": aux["perplexity"].detach(),
        "temperature": aux["temperature"].detach(),
        "bits_per_token_est": bpt.detach(),
    }
    return loss, metrics