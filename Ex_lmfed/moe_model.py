# moe_model.py

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    LlamaModel,
    LlamaConfig,
)

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.A = nn.Parameter(torch.randn(out_features, r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, in_features) * 0.01)

    def forward(self, x):
        W_eff = self.weight + self.A @ self.B
        return x @ W_eff.T + self.bias


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, lora_rank=4):
        super().__init__()
        self.ffn = nn.Sequential(
            LoRALinear(input_dim, hidden_dim, r=lora_rank),
            nn.ReLU(),
            LoRALinear(hidden_dim, input_dim, r=lora_rank)
        )

    def forward(self, x):
        return self.ffn(x)


class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, top_k=1, lora_rank=4):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, lora_rank=lora_rank) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gate_logits = self.gate(x)
        weights = torch.softmax(gate_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)

        output = 0
        for i in range(self.top_k):
            idx = topk_indices[:, i]
            weight = topk_weights[:, i].unsqueeze(-1)
            expert_out = self.experts[idx[0]](x)
            output += expert_out * weight

        return output, topk_indices


class LMFedMoEModel(nn.Module):
    def __init__(self,
                 model_type="bert",
                 model_name="bert-base-uncased",
                 moe_hidden_dim=512,
                 num_experts=4,
                 lora_rank=4,
                 num_classes=2):
        super().__init__()
        self.model_type = model_type.lower()

        if self.model_type == "bert":
            self.encoder = AutoModel.from_pretrained(model_name)
            encoder_hidden = self.encoder.config.hidden_size
        elif self.model_type == "llama":
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=config)
            encoder_hidden = self.encoder.config.hidden_size
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.moe = MoELayer(
            input_dim=encoder_hidden,
            hidden_dim=moe_hidden_dim,
            num_experts=num_experts,
            top_k=1,
            lora_rank=lora_rank
        )
        self.classifier = nn.Linear(encoder_hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0]
        moe_out, expert_ids = self.moe(cls_token)
        logits = self.classifier(moe_out)
        return logits, expert_ids
