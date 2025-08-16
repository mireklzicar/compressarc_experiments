import torch
from transformers import T5Config
from models.vit_arc.modeling import ViTARCForConditionalGeneration

class CompressionViTARC:
    def __init__(self, task):
        self.task = task
        self.config = T5Config.from_pretrained('t5-small')
        self.config.use_OPE = False
        
        # Add required attributes for ViTARC model
        self.config.rows = 33  # Default grid height for ARC
        self.config.cols = 34  # Default grid width for ARC
        
        self.model = ViTARCForConditionalGeneration(self.config)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits