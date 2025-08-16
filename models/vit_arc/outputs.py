from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class LatentPromptSeq2SeqLMOutput(Seq2SeqLMOutput):
    latent_logits: Optional[torch.FloatTensor] = None
    latent_embeds: Optional[torch.FloatTensor] = None