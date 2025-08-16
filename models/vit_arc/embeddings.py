import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTARCEmbedding(nn.Module):
    """
    A module for mixing input embeddings and positional embeddings according to different strategies.
    Instead of reading from a config, it now directly takes a `mixer_strategy` argument.

    Supported strategies:
      - 'hardcoded_normalization'
      - 'learnable_scaling'
      - 'weighted_sum'
      - 'weighted_sum_no_norm'
      - 'learnable_scaling_vec'
      - 'weighted_sum_vec'
      - 'weighted_sum_no_norm_vec'
      - 'positional_attention'
      - 'layer_norm'
      - 'default'

    Example usage:
        embedding_module = ViTARCEmbedding(embed_dim=512, mixer_strategy='weighted_sum')
        output_embeds = embedding_module(inputs_embeds, position_embeds)
    """
    def __init__(self, embed_dim: int, mixer_strategy: str):
        super().__init__()
        self.embed_dim = embed_dim
        self.mixer_strategy = mixer_strategy

        # For 'learnable_scaling_vec', 'weighted_sum_vec', 'weighted_sum_no_norm_vec'
        # we need vector-based parameters (1, embed_dim).
        if self.mixer_strategy in ['learnable_scaling_vec',
                                   'weighted_sum_vec',
                                   'weighted_sum_no_norm_vec']:
            self.position_scale = nn.Parameter(torch.ones(1, embed_dim))
            self.input_weight = nn.Parameter(torch.ones(1, embed_dim))
            self.position_weight = nn.Parameter(torch.ones(1, embed_dim))

        # For 'learnable_scaling', 'weighted_sum', 'weighted_sum_no_norm'
        # we need scalar-based parameters (1,).
        if self.mixer_strategy in ['learnable_scaling',
                                   'weighted_sum',
                                   'weighted_sum_no_norm']:
            self.position_scale = nn.Parameter(torch.ones(1))
            self.input_weight = nn.Parameter(torch.ones(1))
            self.position_weight = nn.Parameter(torch.ones(1))

        # For 'positional_attention'
        if self.mixer_strategy == 'positional_attention':
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)

        # For 'layer_norm'
        if self.mixer_strategy == 'layer_norm':
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs_embeds: torch.Tensor, position_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]
            position_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]

        Returns:
            output_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]
        """
        strategy = self.mixer_strategy

        if strategy == 'hardcoded_normalization':
            inputs_embeds_norm = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds = inputs_embeds_norm + position_embeds_norm

        elif strategy in ['learnable_scaling', 'learnable_scaling_vec']:
            scaled_position_embeds = self.position_scale * position_embeds
            output_embeds = inputs_embeds + scaled_position_embeds

        elif strategy in ['weighted_sum', 'weighted_sum_vec']:
            inputs_embeds_norm = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds = (self.input_weight * inputs_embeds_norm) + (self.position_weight * position_embeds_norm)

        elif strategy in ['weighted_sum_no_norm', 'weighted_sum_no_norm_vec']:
            output_embeds = (self.input_weight * inputs_embeds) + (self.position_weight * position_embeds)

        elif strategy == 'positional_attention':
            # Expand position_embeds to match the batch dimension of inputs_embeds
            position_embeds_expanded = position_embeds.expand(inputs_embeds.shape[0], -1, -1)

            # Reshape to [seq_len, batch_size, embed_dim] for MultiheadAttention
            inputs_embeds_reshaped = inputs_embeds.transpose(0, 1)
            position_embeds_reshaped = position_embeds_expanded.transpose(0, 1)

            attn_output, _ = self.attention(
                inputs_embeds_reshaped,
                position_embeds_reshaped,
                position_embeds_reshaped
            )
            output_embeds = inputs_embeds_reshaped + attn_output
            output_embeds = output_embeds.transpose(0, 1)  # back to [batch_size, seq_len, embed_dim]

        elif strategy == 'layer_norm':
            combined_embeds = inputs_embeds + position_embeds
            output_embeds = self.layer_norm(combined_embeds)

        elif strategy == 'default':
            output_embeds = inputs_embeds + position_embeds

        else:
            raise ValueError(f"Unsupported mixer_strategy: {strategy}")

        return output_embeds


# SinusoidalAPE 
class FixedAbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(16384).type_as(inv_freq)
        sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.embed = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids: torch.Tensor):
        return self.embed(position_ids.long())