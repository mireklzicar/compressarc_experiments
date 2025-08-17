#!/usr/bin/env python3
"""
Test script to verify that the spatial-aware loss function correctly penalizes
size mismatches in ARC grid predictions.
"""

import torch
import numpy as np
from tokenization.arc_tokenizer import get_or_build_arc_tokenizer
from models.vit_arc.train_step import compute_loss, spatial_reconstruction_loss

def create_test_case():
    """Create a test case where predicted grid is smaller than target."""
    tokenizer = get_or_build_arc_tokenizer()
    
    # Target: 3x3 grid
    target_grid_tokens = "<arc_1><arc_2><arc_3><arc_nl><arc_4><arc_5><arc_6><arc_nl><arc_7><arc_8><arc_9>"
    target_token_ids = tokenizer.encode(target_grid_tokens)
    
    # Prediction: 2x2 grid (smaller - the problematic behavior we want to fix)
    pred_grid_tokens = "<arc_1><arc_2><arc_nl><arc_4><arc_5>"
    pred_token_ids = tokenizer.encode(pred_grid_tokens)
    
    # Convert to tensors and pad to same length for comparison
    max_len = max(len(target_token_ids), len(pred_token_ids))
    
    target_padded = target_token_ids + [tokenizer.pad_token_id] * (max_len - len(target_token_ids))
    pred_padded = pred_token_ids + [tokenizer.pad_token_id] * (max_len - len(pred_token_ids))
    
    # Create fake logits (one-hot for predicted tokens)
    vocab_size = len(tokenizer)
    pred_logits = torch.zeros(1, max_len, vocab_size)
    for i, token_id in enumerate(pred_padded):
        pred_logits[0, i, token_id] = 10.0  # High confidence
    
    target_tensor = torch.tensor([target_padded], dtype=torch.long)
    
    return pred_logits, target_tensor, tokenizer

def test_spatial_vs_regular_loss():
    """Test that spatial loss penalizes size mismatch more than regular loss."""
    
    print("Testing Spatial-Aware Loss vs Regular Cross-Entropy Loss")
    print("=" * 60)
    
    pred_logits, target_tensor, tokenizer = create_test_case()
    
    # Create dummy aux data
    aux = {
        "kl_uniform": torch.tensor(0.1),
        "entropy": torch.tensor(0.05),
        "probs": torch.rand(1, pred_logits.shape[1], 512),  # Dummy probs
        "perplexity": torch.tensor(2.0),
        "temperature": torch.tensor(1.0)
    }
    
    # Test regular loss
    print("1. Regular Cross-Entropy Loss:")
    regular_loss, regular_metrics = compute_loss(
        pred_logits, target_tensor, aux, 
        tokenizer=tokenizer, 
        use_spatial_loss=False, 
        w_spatial=1.0
    )
    print(f"   Total Loss: {regular_loss:.4f}")
    print(f"   Reconstruction: {regular_metrics['loss_rec']:.4f}")
    print(f"   Size Penalty: {regular_metrics['loss_size']:.4f}")
    
    # Test spatial loss
    print("\n2. Spatial-Aware Loss:")
    spatial_loss, spatial_metrics = compute_loss(
        pred_logits, target_tensor, aux,
        tokenizer=tokenizer,
        use_spatial_loss=True,
        w_spatial=10.0  # Strong spatial penalty
    )
    print(f"   Total Loss: {spatial_loss:.4f}")
    print(f"   Reconstruction: {spatial_metrics['loss_rec']:.4f}")
    print(f"   Spatial Penalty: {spatial_metrics['loss_spatial']:.4f}")
    
    # Test direct spatial reconstruction loss
    print("\n3. Direct Spatial Reconstruction Loss:")
    direct_spatial = spatial_reconstruction_loss(pred_logits, target_tensor, tokenizer, size_penalty_weight=10.0)
    print(f"   Direct Loss: {direct_spatial:.4f}")
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"Spatial loss should be significantly higher: {spatial_loss:.4f} > {regular_loss:.4f}")
    if spatial_loss > regular_loss:
        print("✅ SUCCESS: Spatial loss correctly penalizes size mismatch!")
    else:
        print("❌ FAILURE: Spatial loss not working as expected")
    
    return spatial_loss > regular_loss

if __name__ == "__main__":
    success = test_spatial_vs_regular_loss()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")