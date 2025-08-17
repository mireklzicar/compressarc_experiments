import torch
import torch.nn as nn
from .gumbel_bottleneck import GumbelSoftmaxBottleneck

class ViTARCWithGumbel(nn.Module):
    def __init__(self, encoder, decoder, lm_head, embed_dim, codebook_size=512, **bottleneck_kwargs):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = GumbelSoftmaxBottleneck(
            embed_dim=embed_dim, codebook_size=codebook_size, **bottleneck_kwargs
        )
        self.decoder = decoder
        self.lm_head = lm_head

    def forward(self, x, temp: float = None, hard: bool = None, return_aux: bool = False):
        """
        Expect `encoder(x)` to return token embeddings (B, N, D).
        If your encoder returns a dict, adapt below accordingly.
        """
        encoder_output = self.encoder(x)             # Returns BaseModelOutputWithPastAndCrossAttentions
        tokens = encoder_output.last_hidden_state    # Extract the actual tensor (B, N, D)
        
        #print(f"DEBUG ViTARCWithGumbel: encoder output shape = {tokens.shape}")
        
        z, aux = self.bottleneck(tokens, temp=temp, hard=hard)
        # Pass continuous quantized embeddings to decoder, not discrete indices
        indices = aux["indices"]                     # (B, N) discrete token indices
        
        #print(f"DEBUG ViTARCWithGumbel: bottleneck indices shape = {indices.shape}")
        #print(f"DEBUG ViTARCWithGumbel: bottleneck z shape = {z.shape}")
        
        decoder_output = self.decoder(inputs_embeds=z)       # Returns BaseModelOutputWithPastAndCrossAttentions
        hidden_states = decoder_output.last_hidden_state       # Extract the actual tensor
        
        #print(f"DEBUG ViTARCWithGumbel: decoder hidden_states shape = {hidden_states.shape}")
        
        # Convert embeddings to vocabulary logits using lm_head
        logits = self.lm_head(hidden_states)          # (B, N, vocab_size)
        
        #print(f"DEBUG ViTARCWithGumbel: final logits shape = {logits.shape}")
        
        return (logits, aux) if return_aux else logits