import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Block, T5LayerFF, T5Stack

from .attention import CustomT5LayerCrossAttention, CustomT5LayerSelfAttention
from .embeddings import FixedAbsolutePositionalEmbedding, ViTARCEmbedding

logger = logging.getLogger("debug")


class CustomT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)        
        # Defaults if not present in config
        self.ape_type = getattr(config, "ape_type", "SinusoidalAPE2D")
        self.rpe_type = getattr(config, "rpe_type", "Two-slope-Alibi")
        self.rpe_abs = getattr(config, "rpe_abs", True)
        self.use_OPE = getattr(config, "use_OPE", True)
        self.ape_mixer_strategy = getattr(config, "ape_mixer", "default")

        self.layer = nn.ModuleList()
        self.layer.append(CustomT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(CustomT5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,        
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        relative_position=None,        
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (key / value) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            relative_position=relative_position,            
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,                
                relative_position=relative_position,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)                

        # Defaults if not present in config
        self.ape_type = getattr(config, "ape_type", "SinusoidalAPE2D")
        self.rpe_type = getattr(config, "rpe_type", "Two-slope-Alibi")
        self.rpe_abs = getattr(config, "rpe_abs", True)
        self.use_OPE = getattr(config, "use_OPE", True)
        self.ape_mixer_strategy = getattr(config, "ape_mixer", "default")
        self.num_grids = getattr(config, "num_grids", 1)  # Default to 1 grid

        self.block = nn.ModuleList(
            [CustomT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )

        self.APE_mixer = ViTARCEmbedding(config.d_model, self.ape_mixer_strategy)
        self.config = config

        if self.ape_type == "LearnedAPE":
            # 2D LearnedAPE is the same as LearnedAPE
            # 2D LearnedAPE + OPE is not implemented, but you can extend base on the code easily
            self.wpe = nn.Embedding(2048, config.d_model)
            self.wpe.weight.data.normal_(
                    mean=0.0, std=config.initializer_factor * 1.0
            )

        elif self.ape_type == "SinusoidalAPE":
            self.wpe = FixedAbsolutePositionalEmbedding(config.d_model)            

        elif self.ape_type == "SinusoidalAPE2D":            
            # 2D APE for encoder and cross attn            
            if config.use_OPE:
                # If with OPE, half enc is reserved for obj_idx
                self.wpe_obj_enc = FixedAbsolutePositionalEmbedding(config.d_model/2) # 128/2 -> 64
                self.wpe_x_enc = FixedAbsolutePositionalEmbedding(config.d_model/4) # 128/4 -> 32
                self.wpe_y_enc = FixedAbsolutePositionalEmbedding(config.d_model/4) # 128/4 -> 32

            # Decoder is the same old 2D
            self.wpe_x = FixedAbsolutePositionalEmbedding(config.d_model/2) # 128/2 -> 64
            self.wpe_y = FixedAbsolutePositionalEmbedding(config.d_model/2) # 128/2 -> 64

            # 1D APE for decoder/ non-2d positions
            self.wpe = FixedAbsolutePositionalEmbedding(config.d_model)

    def _create_dynamic_rpe_matrix(self, grid_height, grid_width, num_grids, device):
        """
        Dynamically creates the RPE matrix. For multi-grid inputs, it creates a block-diagonal matrix.
        Caches the result to avoid re-computation.
        """
        if not hasattr(self, '_rpe_cache'):
            self._rpe_cache = {}

        cache_key = (grid_height, grid_width, num_grids)
        if cache_key in self._rpe_cache:
            return self._rpe_cache[cache_key].to(device)
            
        large_dist = grid_width + 10
        single_grid_relative_pos = self.calculate_2d_relative_positions(grid_height, grid_width)
        
        # Base matrix for one grid, including special tokens
        grid_len_no_special = grid_height * grid_width
        grid_len_with_special = grid_len_no_special + 2
        
        single_dist_matrix = torch.full((grid_len_with_special, grid_len_with_special), fill_value=large_dist, dtype=torch.float)
        single_dist_matrix[1:1 + grid_len_no_special, 1:1 + grid_len_no_special] = single_grid_relative_pos
        single_dist_matrix[0, :] = large_dist
        single_dist_matrix[:, 0] = large_dist
        single_dist_matrix[-1, :] = large_dist + 1
        single_dist_matrix[:, -1] = large_dist + 1

        if num_grids <= 1:
            full_matrix = single_dist_matrix
        else:
            # For multi-grid, create a block-diagonal matrix
            block_matrices = [single_dist_matrix] * num_grids
            full_matrix = torch.block_diag(*block_matrices)
            
            # Fill off-diagonal blocks with large distance values
            for i in range(num_grids):
                for j in range(num_grids):
                    if i != j:
                        row_start, row_end = i * grid_len_with_special, (i + 1) * grid_len_with_special
                        col_start, col_end = j * grid_len_with_special, (j + 1) * grid_len_with_special
                        full_matrix[row_start:row_end, col_start:col_end] = large_dist

        self._rpe_cache[cache_key] = full_matrix.cpu()
        return full_matrix.to(device)

    def calculate_2d_relative_positions(self, grid_height, grid_width):
        if self.rpe_type == "Four-diag-slope-Alibi":
            # Define direction-specific factors
            # Pre-mult those to diagonal directions
            top_right_factor = 2 ** 0.25
            down_right_factor = 2 ** 0.25
        else:
            top_right_factor = 1.0
            down_right_factor = 1.0
        

        # Create grid coordinates
        x_coords, y_coords = torch.meshgrid(
            torch.arange(grid_height, dtype=torch.long),
            torch.arange(grid_width, dtype=torch.long),
            indexing='ij'
        )

        # Flatten the 2D grid coordinates
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()

        # Initialize the relative position matrix
        num_positions = grid_height * grid_width
        
        # Vectorized calculation of Manhattan distances
        x_diff = x_flat.unsqueeze(0) - x_flat.unsqueeze(1)  # Shape: (num_positions, num_positions)
        y_diff = y_flat.unsqueeze(0) - y_flat.unsqueeze(1)  # Shape: (num_positions, num_positions)
        manhattan_distance = (torch.abs(x_diff) + torch.abs(y_diff)).float()
        
        # Apply direction-specific factors
        if self.rpe_type == "Four-diag-slope-Alibi":
            # Top-right: x_diff < 0 and y_diff < 0
            top_right_mask = (x_diff < 0) & (y_diff < 0)
            manhattan_distance = torch.where(top_right_mask, manhattan_distance * top_right_factor, manhattan_distance)
            
            # Down-right: x_diff > 0 and y_diff < 0
            down_right_mask = (x_diff > 0) & (y_diff < 0)
            manhattan_distance = torch.where(down_right_mask, manhattan_distance * down_right_factor, manhattan_distance)

        return manhattan_distance


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        position_ids=None,
        return_dict=None,
        relative_position=None,
        object_idx=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        
        if self.rpe_type in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
            relative_position = self._create_dynamic_rpe_matrix(
                grid_height=self.config.rows,
                grid_width=self.config.cols,
                num_grids=self.num_grids,
                device=inputs_embeds.device
            )

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length        

        if self.ape_type in ["SinusoidalAPE2D"]:
            # 1) Prepare or shape position_ids
            if position_ids is not None:
                position_ids = position_ids.view(-1, input_shape[-1])

            if past_key_values is None:
                past_length = 0
            else:
                # Usually from self-attn: past_key_values[0] => (k, v) with shape [batch, n_heads, seq_len, dim_per_head]
                # so we take the -2 dimension for length
                past_length = past_key_values[0][0].size(-2)

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    input_shape[-1] + past_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

            # 2) Build the core 2D embeddings
            #    - rows/cols stored in config => e.g. ARC shapes
            rows = getattr(self.config, "rows", 33)
            cols = getattr(self.config, "cols", 34)
            grid_len = rows * cols

            # Flatten position_ids if needed
            flat_position_ids = position_ids.view(-1)

            # X-coordinates => repeated row times, Y-coordinates => repeat_interleave col times
            position_ids_x = torch.arange(cols, device=device).repeat(rows)
            position_ids_y = torch.arange(rows, device=device).repeat_interleave(cols)

            # Repeat for the batch
            pos_batch_size = position_ids.shape[0]
            position_ids_x = position_ids_x.repeat(pos_batch_size, 1)
            position_ids_y = position_ids_y.repeat(pos_batch_size, 1)

            # Build your 2D embeddings
            # object_idx-based embeddings only in encoder
            if self.use_OPE and (not self.is_decoder) and object_idx is not None:
               # Tetrominoes case: The input is three concatenated grids.
               # We need to construct positional embeddings that match this structure.

               # 1. Create X/Y pos_ids for a single padded grid (33*34=1122), padded to 1124.
               pos_ids_x_single = torch.arange(cols, device=device).repeat(rows)
               pos_ids_y_single = torch.arange(rows, device=device).repeat_interleave(cols)
               pos_ids_x_padded = F.pad(pos_ids_x_single, (1, 1), 'constant', 0)
               pos_ids_y_padded = F.pad(pos_ids_y_single, (1, 1), 'constant', 0)

               # 2. Concatenate them three times to match the input structure (I1, O1, I2)
               concatenated_pos_x = torch.cat([pos_ids_x_padded, pos_ids_x_padded, pos_ids_x_padded], dim=0)
               concatenated_pos_y = torch.cat([pos_ids_y_padded, pos_ids_y_padded, pos_ids_y_padded], dim=0)
               
               # 3. Pad the concatenated pos_ids to match the object_idx length
               final_len = object_idx[:, 1:-1].shape[1]
               final_pos_x = torch.zeros(final_len, dtype=torch.long, device=device)
               final_pos_y = torch.zeros(final_len, dtype=torch.long, device=device)
               len_to_copy = min(len(concatenated_pos_x), final_len)
               final_pos_x[:len_to_copy] = concatenated_pos_x[:len_to_copy]
               final_pos_y[:len_to_copy] = concatenated_pos_y[:len_to_copy]

               # 4. Create embeddings and ensure correct batch size
               object_embeds = self.wpe_obj_enc(object_idx[:, 1:-1])
               
               # Create position embeddings for a single example, then expand
               final_pos_x_single = final_pos_x.unsqueeze(0)  # [1, seq_len]
               final_pos_y_single = final_pos_y.unsqueeze(0)  # [1, seq_len]
               
               position_embeds_x = self.wpe_x_enc(final_pos_x_single)  # [1, seq_len, embed_dim]
               position_embeds_y = self.wpe_y_enc(final_pos_y_single)  # [1, seq_len, embed_dim]
               
               # Expand all to match batch size
               if object_embeds.shape[0] != batch_size:
                   object_embeds = object_embeds.expand(batch_size, -1, -1)
               if position_embeds_x.shape[0] != batch_size:
                   position_embeds_x = position_embeds_x.expand(batch_size, -1, -1)
               if position_embeds_y.shape[0] != batch_size:
                   position_embeds_y = position_embeds_y.expand(batch_size, -1, -1)
                   
               # 5. Combine embeddings
               position_embeds_2d = torch.cat((object_embeds, position_embeds_x, position_embeds_y), dim=-1)

            else:
                # Original (re_arc) or decoder case: Normal 2D scenario for a single grid
                position_embeds_x = self.wpe_x(position_ids_x)
                position_embeds_y = self.wpe_y(position_ids_y)
                position_embeds_x = position_embeds_x.expand(batch_size, -1, -1)
                position_embeds_y = position_embeds_y.expand(batch_size, -1, -1)
                position_embeds_2d = torch.cat((position_embeds_x, position_embeds_y), dim=-1)

            # Also build 1D embeddings (some fallback for tokens outside the 2D region)
            position_embeds_1d = self.wpe(position_ids)  # shape [batch, seq_len, embed_dim]
            position_embeds_1d = position_embeds_1d.expand(position_embeds_2d.size(0), -1, -1)  # Expand along the batch size

            # 3) Insert 2D portion into 1D
            # We store final in 'position_embeds'
            # We'll typically place 2D from index=1 up to grid_len - 1, etc.
            p_seq_len = position_ids.shape[-1]
            position_embeds = position_embeds_1d.clone()
            # print("batch_size",batch_size)
            # print("position_embeds.shape", position_embeds.shape)
            # print("position_embeds_2d.shape", position_embeds_2d.shape)
            

            # A) If is_decoder => we handle offsets differently
            if self.is_decoder:
                # For the decoder, we often have the first token as <pad> or <s>.
                # We'll place the 2D portion from index [1 : grid_len-1] if enough length
                if p_seq_len >= grid_len - 1:
                    position_embeds[:, 1 : grid_len - 1] = position_embeds_2d[:, : grid_len - 2]
                elif p_seq_len == 1:
                    # Possibly only the first token <s> or <pad>, do nothing or partial
                    pos_index = flat_position_ids[0]
                    if pos_index == 0:
                        # e.g. first token is <s>, no 2D portion
                        pass
                    elif 1 <= pos_index < (grid_len - 2):
                        # place that single token from position_embeds_2d
                        # e.g. position_embeds[:, 0] = position_embeds_2d[:, pos_index-1]
                        position_embeds[:, 0] = position_embeds_2d[:, pos_index - 1]
                    else:
                        # pos_index beyond 2D range => fallback
                        pass
                else:
                    # partial coverage: we have p_seq_len > 1 but less than grid_len-1
                    available_2d_len = position_embeds_2d.shape[1]
                    target_len = min(p_seq_len - 1, available_2d_len)
                    position_embeds[:, 1 : 1 + target_len] = position_embeds_2d[:, :target_len]

            else:
                # B) If not is_decoder => simpler approach for an encoder or partial usage
                # We might do something similar: fill [1 : grid_len-1]
                if p_seq_len >= grid_len:
                    position_embeds[:, 1 : grid_len] = position_embeds_2d[:, : (grid_len - 1)]
                else:
                    # partial coverage
                    available_2d_len = position_embeds_2d.shape[1]
                    target_len = min(p_seq_len - 1, available_2d_len)
                    position_embeds[:, 1 : 1 + target_len] = position_embeds_2d[:, :target_len]

            # 4) Finally mix them into inputs_embeds using APE_mixer
            inputs_embeds = self.APE_mixer(inputs_embeds, position_embeds)
        

        if self.ape_type in [
            "SinusoidalAPE",
            "LearnedAPE",
        ]:
            # 1D APE cases
            if position_ids is not None:
                position_ids = position_ids.view(-1, input_shape[-1])

            if past_key_values is None:
                past_length = 0
            else:
                past_length = past_key_values[0][0].size(-2)

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    input_shape[-1] + past_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            
            position_embeds = self.wpe(position_ids)   
            inputs_embeds = self.APE_mixer(inputs_embeds, position_embeds)         
            #inputs_embeds += position_embeds

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None        
        encoder_decoder_position_bias = None
        

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)                
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)                
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,                    
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,                    
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    relative_position=relative_position,  # Pass the relative_position to the layer
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights),
            #                                  (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]            
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]                

            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)


            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )