import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

from .blocks import CustomT5Stack
from .outputs import LatentPromptSeq2SeqLMOutput


class ViTARCForConditionalGeneration(T5ForConditionalGeneration):
    """
    Specialized T5-based model for the ViTARC project, extending T5ForConditionalGeneration.

    This model can read the following fields from the T5Config (if present):
      - ape_type (str): e.g. 'SinusoidalAPE', 'SinusoidalAPE2D', 'LearnedAPE', or 'none'. Defaults to 'SinusoidalAPE2D'.
      - rpe_type (str): e.g. 'Four-diag-slope-Alibi','Two-slope-Alibi'. Defaults to 'Two-slope-Alibi'.
      - rpe_abs (bool): default True or False if not present.
      - use_OPE (bool): default True.
      - ape_mixer (str): indicates the approach to mixing embeddings, e.g. 'learnable_scaling', 'weighted_sum', etc.
                         (not used in this snippet, just carried in config).
    
    """
    def __init__(self, config: T5Config):        
        """
        Extracts custom positional-encoding fields from config if available:
          ape_type, rpe_type, rpe_abs, use_OPE, ape_mixer.
        """
        # Defaults if not present in config
        self.ape_type = getattr(config, "ape_type", "SinusoidalAPE2D")
        self.rpe_type = getattr(config, "rpe_type", "Two-slope-Alibi")
        self.rpe_abs = getattr(config, "rpe_abs", True)
        self.use_OPE = getattr(config, "use_OPE", True)
        self.ape_mixer_strategy = getattr(config, "ape_mixer", "default")

        super().__init__(config)
        self.model_dim = config.d_model        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = CustomT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,               
        object_idx: Optional[torch.FloatTensor] = None,   
        **kwargs      # To ignore new HF transformer params like cache_position       
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                #warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,                
                object_idx=object_idx,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class LatentPromptViTARC(ViTARCForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.latent_vocab_size = getattr(config, "latent_vocab_size", 16)
        self.num_latent_tokens = getattr(config, "num_latent_tokens", 10)
        
        self.latent_head = nn.Linear(config.d_model, self.latent_vocab_size)
        self.latent_embeds_projection = nn.Linear(self.latent_vocab_size, config.d_model, bias=False)

    def forward(
        self,
        input_ids_t1: Optional[torch.LongTensor] = None,
        attention_mask_t1: Optional[torch.FloatTensor] = None,
        input_ids_t2: Optional[torch.LongTensor] = None,
        attention_mask_t2: Optional[torch.FloatTensor] = None,
        labels_t2: Optional[torch.LongTensor] = None,
        # Arguments for Hugging Face compatibility
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Inference path (called from generate)
        if encoder_outputs is not None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                labels=labels,
                return_dict=return_dict,
                **kwargs,
            )

        # Training path
        # T1: Generate latent prompt by running a full encode-decode pass on the example pair.
        # We provide input_ids_t1 as "dummy" labels to force the decoder to run. The resulting loss is ignored.
        t1_outputs = super().forward(
            input_ids=input_ids_t1,
            attention_mask=attention_mask_t1,
            labels=input_ids_t1,
            output_hidden_states=True,
            return_dict=True,
        )
        t1_decoder_hidden_states = t1_outputs.decoder_hidden_states[-1]
        
        latent_logits = self.latent_head(t1_decoder_hidden_states[:, :self.num_latent_tokens, :])
        latent_tokens_one_hot = F.gumbel_softmax(latent_logits, tau=1.0, hard=True, dim=-1)
        latent_embeds = self.latent_embeds_projection(latent_tokens_one_hot)

        # T2: Predict output2
        t2_encoder_outputs = self.encoder(input_ids=input_ids_t2, attention_mask=attention_mask_t2)
        encoder_hidden_states = t2_encoder_outputs[0]
        
        combined_encoder_hidden_states = torch.cat([latent_embeds, encoder_hidden_states], dim=1)
        latent_attention_mask = torch.ones(latent_embeds.shape[:2], device=attention_mask_t2.device)
        combined_attention_mask = torch.cat([latent_attention_mask, attention_mask_t2], dim=1)

        final_labels = labels if labels is not None else labels_t2
        
        if final_labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(final_labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=combined_encoder_hidden_states,
            encoder_attention_mask=combined_attention_mask,
        )

        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if final_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), final_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return LatentPromptSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=t2_encoder_outputs.last_hidden_state,
            latent_logits=latent_logits,
            latent_embeds=latent_embeds,
        )

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # This is a standard part of Hugging Face model API for generation
        encoder_outputs = kwargs.get("encoder_outputs")
        
        # if `past_key_values` is passed, we're in the middle of decoding
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # input_ids are not needed when encoder_outputs is provided
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    def generate(self, input_ids_t1, attention_mask_t1, input_ids_t2, attention_mask_t2, **kwargs):
        # Step 1: Generate latent prompt from the first input pair by running a full encode-decode pass.
        # The loss is not used; we only need the decoder's hidden states.
        t1_outputs = super().forward(
            input_ids=input_ids_t1,
            attention_mask=attention_mask_t1,
            labels=input_ids_t1,
            output_hidden_states=True,
            return_dict=True,
        )
        t1_decoder_hidden_states = t1_outputs.decoder_hidden_states[-1]
        
        latent_logits = self.latent_head(t1_decoder_hidden_states[:, :self.num_latent_tokens, :])
        latent_tokens = torch.argmax(latent_logits, dim=-1)
        latent_embeds = self.latent_embeds_projection(F.one_hot(latent_tokens, num_classes=self.latent_vocab_size).float())

        # Step 2: Prepare the combined encoder states for the second input
        encoder_outputs_t2 = self.encoder(input_ids=input_ids_t2, attention_mask=attention_mask_t2)
        encoder_hidden_states_t2 = encoder_outputs_t2[0]
        
        combined_encoder_hidden_states = torch.cat([latent_embeds, encoder_hidden_states_t2], dim=1)
        latent_attention_mask = torch.ones(latent_embeds.shape[:2], device=attention_mask_t2.device)
        combined_attention_mask = torch.cat([latent_attention_mask, attention_mask_t2], dim=1)

        # The `encoder_outputs` are now fixed for the generation process
        encoder_outputs = BaseModelOutput(last_hidden_state=combined_encoder_hidden_states)

        # Step 3: Call the main generate method with the prepared inputs
        return super().generate(
            encoder_outputs=encoder_outputs,
            attention_mask=combined_attention_mask,
            **kwargs,
        )