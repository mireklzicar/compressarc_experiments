import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

def tokens_to_grid_array(tokens, tokenizer):
    """
    Convert a sequence of tokens back to a 2D grid array for loss computation.
    Returns a numpy array representing the grid.
    """
    try:
        # Handle different input formats
        if isinstance(tokens, torch.Tensor):
            if tokens.dim() > 1:
                tokens = tokens[0]  # Take first batch element
            tokens = tokens.tolist()
        
        # Decode tokens to text
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        
        # Clean up special tokens
        cleaned_string = re.sub(r'<s>|</s>|<pad>', '', text).strip()
        
        # Parse grid structure
        if '<arc_nl>' in cleaned_string:
            # Normal case: grid has row separators
            rows = cleaned_string.split('<arc_nl>')
            grid = []
            for row_str in rows:
                if not row_str:
                    continue
                # Find all <arc_DIGIT> tokens in the row
                found_tokens = re.findall(r'<arc_(\d+)>', row_str)
                if found_tokens:
                    row = [int(t) for t in found_tokens]
                    if row:  # Only add non-empty rows
                        grid.append(row)
        else:
            # Fallback: flat sequence - try to reshape into square-ish grid
            all_tokens = re.findall(r'<arc_(\d+)>', cleaned_string)
            if all_tokens:
                token_values = [int(t) for t in all_tokens]
                total_tokens = len(token_values)
                
                # Try to infer grid dimensions
                if total_tokens > 0:
                    # Find best square-ish dimensions
                    best_h = int(np.sqrt(total_tokens))
                    if best_h * best_h == total_tokens:
                        h, w = best_h, best_h
                    else:
                        # Find best factorization
                        for h in range(1, min(total_tokens + 1, 21)):
                            if total_tokens % h == 0:
                                w = total_tokens // h
                                if w <= 30:  # Reasonable width limit
                                    break
                        else:
                            h, w = 1, total_tokens  # Fallback to single row
                    
                    grid = []
                    for i in range(h):
                        start_idx = i * w
                        end_idx = start_idx + w
                        if start_idx < total_tokens:
                            row = token_values[start_idx:min(end_idx, total_tokens)]
                            grid.append(row)
                else:
                    grid = [[0]]  # Empty fallback
            else:
                grid = [[0]]  # Empty fallback
        
        if not grid:
            return np.array([[0]], dtype=np.int32)
        
        # Ensure all rows have the same length (pad with zeros)
        max_len = max(len(row) for row in grid) if grid else 1
        padded_grid = []
        for row in grid:
            padded_row = row + [0] * (max_len - len(row))
            padded_grid.append(padded_row)
        
        grid_array = np.array(padded_grid, dtype=np.int32)
        # Clip values to valid ARC range
        grid_array = np.clip(grid_array, 0, 9)
        
        return grid_array
        
    except Exception as e:
        # Return minimal grid on any error
        return np.array([[0]], dtype=np.int32)

def spatial_reconstruction_loss(pred_logits, target_tokens, tokenizer, size_penalty_weight=0.1):
    """
    Compute spatial-aware reconstruction loss that understands 2D grid structure.
    Very gentle penalties to allow model to learn basic token patterns first.
    
    Args:
        pred_logits: (B, N, vocab_size) predicted logits
        target_tokens: (B, N) target token sequence
        tokenizer: tokenizer for converting tokens to grids
        size_penalty_weight: weight for penalizing size mismatches
    
    Returns:
        loss: scalar tensor
    """
    # First compute standard cross-entropy as baseline
    ce_loss = F.cross_entropy(pred_logits.view(-1, pred_logits.size(-1)), target_tokens.view(-1))
    
    batch_size = pred_logits.shape[0]
    spatial_penalty = 0.0
    
    # Convert predicted logits to tokens
    pred_tokens = torch.argmax(pred_logits, dim=-1)  # (B, N)
    
    for b in range(batch_size):
        try:
            # Convert to grids
            pred_grid = tokens_to_grid_array(pred_tokens[b], tokenizer)
            target_grid = tokens_to_grid_array(target_tokens[b], tokenizer)
            
            # Size mismatch penalty - very gentle
            pred_h, pred_w = pred_grid.shape
            target_h, target_w = target_grid.shape
            
            # Use logarithmic penalty to avoid explosion
            h_diff = abs(pred_h - target_h)
            w_diff = abs(pred_w - target_w)
            size_penalty = (torch.log(torch.tensor(1.0 + h_diff)) +
                          torch.log(torch.tensor(1.0 + w_diff))) * 0.1
            
            # Content penalty on overlapping region - also gentle
            min_h = min(pred_h, target_h)
            min_w = min(pred_w, target_w)
            
            content_penalty = 0.0
            if min_h > 0 and min_w > 0:
                pred_overlap = pred_grid[:min_h, :min_w]
                target_overlap = target_grid[:min_h, :min_w]
                
                # Simple mismatch count
                mismatch_count = np.sum(pred_overlap != target_overlap)
                content_penalty = mismatch_count * 0.01  # Very small per-cell penalty
            
            # Small penalty for missing content
            if target_h > pred_h or target_w > pred_w:
                missing_cells = (target_h * target_w) - (min_h * min_w)
                missing_penalty = min(missing_cells * 0.01, 1.0)  # Cap at 1.0
            else:
                missing_penalty = 0.0
            
            spatial_penalty += (size_penalty + content_penalty + missing_penalty)
            
        except Exception as e:
            # Very small error penalty
            spatial_penalty += 0.1
    
    # Combine CE loss with gentle spatial penalty
    total_spatial_penalty = spatial_penalty / batch_size
    total_loss = ce_loss + size_penalty_weight * total_spatial_penalty
    
    # Convert to proper tensor
    return total_loss

def reconstruction_loss(pred, target, kind="l2"):
    if kind == "l1":
        return (pred - target).abs().mean()
    if kind == "l2":
        return ((pred - target) ** 2).mean()
    if kind == "cross_entropy":
        # pred: (B, N, vocab_size), target: (B, N)
        return F.cross_entropy(pred.view(-1, pred.size(-1)), target.view(-1))
    if kind == "spatial_aware":
        # This will be handled separately in compute_loss
        raise NotImplementedError("Use spatial_reconstruction_loss directly")
    raise ValueError(kind)

def compute_loss(out, target, aux, tokenizer=None, w_kl=1e-2, w_entropy=1e-3, w_spatial=1.0, use_spatial_loss=True, maximize_code_usage=True):
    """
    - w_kl encourages global uniform codebook usage (prevents collapse).
    - w_entropy penalizes per-token entropy to harden assignments => compression.
      If you want softer assignments early, linearly ramp this weight from 0.
    - w_spatial controls spatial-aware reconstruction loss that penalizes size mismatches.
    - use_spatial_loss: if True, use spatial-aware loss; if False, use regular cross-entropy
    """
    klu = aux["kl_uniform"]
    ent = aux["entropy"]
    
    # Choose reconstruction loss type
    if use_spatial_loss and tokenizer is not None:
        # Use spatial-aware loss that understands 2D grid structure
        rec = spatial_reconstruction_loss(out, target, tokenizer, size_penalty_weight=w_spatial)
        size_loss = torch.tensor(0.0, device=out.device)  # Size penalty already included in spatial loss
    else:
        # Fallback to regular cross-entropy loss with separate size penalty
        rec = reconstruction_loss(out, target, kind="cross_entropy")
        
        # Penalize sequence length mismatch
        pred_len = torch.tensor(out.shape[1], device=out.device, dtype=torch.float32)
        target_len = torch.tensor(target.shape[1], device=target.device, dtype=torch.float32)
        size_loss = ((pred_len - target_len) ** 2) * w_spatial

    # Encourage uniform usage => minimize KL(avg_probs || uniform)
    loss = rec + w_kl * klu + size_loss

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
        "loss_spatial" if use_spatial_loss else "loss_size": size_loss.detach() if isinstance(size_loss, torch.Tensor) else torch.tensor(size_loss).detach(),
        "kl_uniform": klu.detach(),
        "entropy": ent.detach(),
        "perplexity": aux["perplexity"].detach(),
        "temperature": aux["temperature"].detach(),
        "bits_per_token_est": bpt.detach(),
    }
    return loss, metrics