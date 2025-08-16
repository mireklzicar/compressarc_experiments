import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerCrossAttention, T5LayerSelfAttention


class CustomT5Attention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False, attn_type="self"):
        super().__init__(config)

         # Defaults if not present in config
        self.ape_type = getattr(config, "ape_type", "SinusoidalAPE2D")
        self.rpe_type = getattr(config, "rpe_type", "Two-slope-Alibi")
        self.rpe_abs = getattr(config, "rpe_abs", True)
        self.use_OPE = getattr(config, "use_OPE", True)
        self.ape_mixer_strategy = getattr(config, "ape_mixer", "default")

        self.d_head = config.d_kv
        self.attn_type = attn_type        
        self.has_relative_attention_bias = has_relative_attention_bias

        if self.has_relative_attention_bias:
            # Apply 2D RPE in 1st layers
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            device = self.relative_attention_bias.weight.device

            if self.rpe_type in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
                # Two slopes are sufficient here, since we manipudate the distance matrix with pre-added per-diag-direction ratios.
                self.slopes_l = torch.Tensor(self.get_slopes(self.n_heads, start_exponent=1)).to(device)*-1
                self.slopes_r = torch.Tensor(self.get_slopes(self.n_heads, start_exponent=0.5)).to(device)*-1
            elif self.rpe_type in ["NoRPE"]:
                #self.relative_attention_bias = None  # No positional encoding bias
                pass
            else:
                self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
                    

    # https://github.com/ofirpress/attention_with_linear_biases/issues/5
    def get_slopes(self, n, start_exponent=1):
        def get_geometric_slopes(n, start_exponent):
            start = 2 ** (-start_exponent)  # Starting value 2^(-start_exponent)
            ratio = 2 ** -1  # Halving each step
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n).is_integer():
            return get_geometric_slopes(n, start_exponent)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (get_geometric_slopes(closest_power_of_2, start_exponent) +
                    self.get_slopes(2 * closest_power_of_2, start_exponent)[0::2][:n - closest_power_of_2])    

    def compute_bias(self, query_length, key_length, device=None, relative_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device

        if self.rpe_type in ["NoRPE"]:
            # Zeros
            return torch.zeros((1, self.n_heads, query_length, key_length), device=device)
        elif self.rpe_type in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
            relative_position = relative_position.to(device)

            if self.rpe_abs:
                relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.n_heads, -1,-1)
            else:
                relative_position = relative_position.unsqueeze(0).expand(self.n_heads, -1,-1)

            self.slopes_l = self.slopes_l.to(device)
            self.slopes_r = self.slopes_r.to(device)

            # relative_position is pre-mult with factor 2**0.25 for top-right, down-right
            alibi_left = self.slopes_l.unsqueeze(1).unsqueeze(1) * relative_position
            alibi_right = self.slopes_r.unsqueeze(1).unsqueeze(1) * relative_position

            values = torch.triu(alibi_right) + torch.tril(alibi_left)

            # Slice the relevant part of the bias before reshaping
            values = values[:, :query_length, :key_length]  # Slicing the tensor before reshaping
            
            values = values.view(1, self.n_heads, query_length, key_length)  # shape (1, num_heads, query_length, key_length)

            return values
        else:
            # Zeros
            return torch.zeros((1, self.n_heads, query_length, key_length), device=device)

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        relative_position=None,        
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]


        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]


        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )

        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        attention_output_dict = {}

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9


        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                if self.rpe_type in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
                    position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device, relative_position=relative_position)
                else:                    
                    position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device, relative_position=None)
            
            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:                
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
                            

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias
        
        scores += position_bias_masked

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class CustomT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.SelfAttention = CustomT5Attention(config, has_relative_attention_bias=has_relative_attention_bias, attn_type="self")
        self.is_decoder = config.is_decoder

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        relative_position=None,        
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,            
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            relative_position=relative_position,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class CustomT5LayerCrossAttention(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)        
        self.EncDecAttention = CustomT5Attention(config, has_relative_attention_bias=False, attn_type="cross")
        self.is_decoder = config.is_decoder

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        relative_position=None,        
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            relative_position=relative_position,            
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs