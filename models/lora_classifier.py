import torch
import torch.nn as nn
from transformers import AutoModel

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r):
        super().__init__()
        self.r = r
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, r))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        lora = x @ self.lora_A.t() @ self.lora_B.t()
        return base + lora

class LoRALinearClassifier(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_args["model_name"])
        hidden_size = self.encoder.config.hidden_size
        self.classifier = LoRALinear(hidden_size, model_args["num_classes"], model_args.get("lora_rank", 4))

    def forward(self, **inputs):
        outputs = self.encoder(**inputs)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)
