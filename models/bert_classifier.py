import torch
import torch.nn as nn
from transformers import AutoModel

class BertClassifier(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_args["model_name"])
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, model_args["num_classes"])

    def forward(self, **inputs):
        outputs = self.encoder(**inputs)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits