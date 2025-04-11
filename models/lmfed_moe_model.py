import torch
import torch.nn as nn
from transformers import AutoModel


class LMFedMoEModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_args["model_name"])
        self.hidden_dim = self.backbone.config.hidden_size
        self.num_experts = model_args.get("num_experts", 4)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, model_args.get("moe_hidden_dim", 512)),
                nn.ReLU(),
                nn.Linear(model_args.get("moe_hidden_dim", 512), self.hidden_dim)
            ) for _ in range(self.num_experts)
        ])
        self.gate = nn.Linear(self.hidden_dim, self.num_experts)
        self.classifier = nn.Linear(self.hidden_dim, model_args.get("num_classes", 2))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token

        # Gating
        gate_logits = self.gate(pooled)  # shape: [batch_size, num_experts]
        gate_scores = torch.softmax(gate_logits, dim=-1)  # soft routing

        # Expert aggregation
        expert_outputs = torch.stack([expert(pooled) for expert in self.experts], dim=1)  # [B, E, D]
        moe_output = torch.sum(gate_scores.unsqueeze(-1) * expert_outputs, dim=1)  # [B, D]

        return self.classifier(moe_output)
