#!/usr/bin/env python3
"""
Debug script to examine what tokens the Gumbel ViTARC model is producing
and why the spatial loss is so high.
"""

import torch
import numpy as np
from tokenization.arc_tokenizer import get_or_build_arc_tokenizer
from models.vit_arc.vitarc_with_gumbel import ViTARCWithGumbel
from compression_vitarc import CompressionViTARC
from models.vit_arc.train_step import compute_loss, tokens_to_grid_array
import train
import preprocessing

def debug_model_outputs():
    """Debug what the model is actually producing."""
    print("Debugging Gumbel ViTARC Model Output")
    print("=" * 50)
    
    # Setup
    torch.set_default_device('cuda')
    task_name = '272f95fa'
    task = preprocessing.preprocess_tasks('training', [task_name])[0]
    
    # Load tokenizer
    tokenizer = get_or_build_arc_tokenizer()
    print(f"Vocab size: {len(tokenizer)}")
    print(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    # Create model with vocabulary fix
    base_model = CompressionViTARC(task)
    
    # Fix vocabulary size mismatch - replace the lm_head with correct vocab size
    arc_vocab_size = len(tokenizer)
    base_model.model.lm_head = torch.nn.Linear(
        base_model.model.config.d_model,
        arc_vocab_size,
        bias=False
    )
    base_model.model.config.vocab_size = arc_vocab_size
    
    model = ViTARCWithGumbel(
        encoder=base_model.model.encoder,
        decoder=base_model.model.decoder,
        lm_head=base_model.model.lm_head,  # Use the corrected lm_head
        embed_dim=512,
        codebook_size=512,
        temp_init=1.0,
        temp_min=0.2,
        anneal_rate=1e-6,
        use_straight_through=True
    )
    print(f"Model lm_head output size: {model.lm_head.out_features}")
    
    # Prepare batch
    batch = train.prepare_vitarc_batch(task, tokenizer)
    input_ids = batch['input_ids'].to(torch.get_default_device())
    attention_mask = batch['attention_mask'].to(torch.get_default_device())
    labels = batch['labels'].to(torch.get_default_device())
    
    print(f"Input batch shape: {input_ids.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print()
    
    # Examine input and target tokens
    print("INPUT EXAMPLES (first training sample):")
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print(f"Input text: {input_text[:200]}...")  # First 200 chars
    print()
    
    print("TARGET EXAMPLES (first training sample):")
    target_text = tokenizer.decode(labels[0], skip_special_tokens=False)
    print(f"Target text: {target_text[:200]}...")  # First 200 chars
    target_grid = tokens_to_grid_array(labels[0], tokenizer)
    print(f"Target grid shape: {target_grid.shape}")
    print(f"Target grid:\n{target_grid}")
    print()
    
    # Forward pass
    print("FORWARD PASS:")
    with torch.no_grad():
        outputs, aux = model(input_ids, return_aux=True)
        
    print(f"Output shape: {outputs.shape}")
    print(f"Output logits range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    
    # Convert to predicted tokens
    pred_tokens = torch.argmax(outputs, dim=-1)
    print(f"Predicted tokens shape: {pred_tokens.shape}")
    
    # Examine predicted tokens for first training sample
    print("\nPREDICTED TOKENS (first training sample):")
    pred_text = tokenizer.decode(pred_tokens[0], skip_special_tokens=False)
    print(f"Predicted text: {pred_text[:200]}...")
    pred_grid = tokens_to_grid_array(pred_tokens[0], tokenizer)
    print(f"Predicted grid shape: {pred_grid.shape}")
    print(f"Predicted grid:\n{pred_grid}")
    print()
    
    # Check for common failure modes
    print("FAILURE MODE ANALYSIS:")
    
    # 1. Check if model is producing mostly pad tokens
    pred_tokens_flat = pred_tokens[0].cpu().numpy()
    pad_count = np.sum(pred_tokens_flat == tokenizer.pad_token_id)
    print(f"PAD token count in prediction: {pad_count}/{len(pred_tokens_flat)} ({100*pad_count/len(pred_tokens_flat):.1f}%)")
    
    # 2. Check token distribution
    unique_tokens, counts = np.unique(pred_tokens_flat, return_counts=True)
    print(f"Unique tokens in prediction: {len(unique_tokens)}")
    print("Top 5 most frequent tokens:")
    for i in np.argsort(counts)[-5:]:
        token_id = unique_tokens[i]
        count = counts[i]
        token_text = tokenizer.decode([token_id])
        print(f"  Token {token_id} ('{token_text}'): {count} times")
    
    # 3. Check if ARC tokens are being generated
    arc_token_pattern = r'<arc_\d+>'
    import re
    arc_matches = re.findall(arc_token_pattern, pred_text)
    print(f"ARC tokens found in prediction: {len(arc_matches)}")
    
    # 4. Compute loss
    print("\nLOSS COMPUTATION:")
    try:
        outputs_train = outputs[:task.n_train]
        labels_train = labels[:task.n_train]
        aux_train = {}
        for k, v in aux.items():
            if isinstance(v, torch.Tensor) and len(v.shape) > 0 and v.shape[0] == outputs.shape[0]:
                aux_train[k] = v[:task.n_train]
            else:
                aux_train[k] = v
                
        loss, metrics = compute_loss(
            outputs_train, labels_train, aux_train,
            tokenizer=tokenizer,
            w_entropy=0.0,
            w_spatial=10.0,
            use_spatial_loss=True
        )
        
        print(f"Total loss: {loss:.4f}")
        print(f"Reconstruction error: {metrics['loss_rec']:.4f}")
        print(f"KL uniform: {metrics['kl_uniform']:.4f}")
        print(f"Spatial loss: {metrics['loss_spatial']:.4f}")
    except Exception as e:
        print(f"Loss computation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_outputs()