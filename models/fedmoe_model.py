import torch
import torch.nn as nn
from transformers import AutoModel

class FedMoEModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_args["model_name"])
        hidden_size = self.encoder.config.hidden_size

        # MoE 구성
        self.num_experts = model_args.get("num_experts", 4)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, model_args["moe_hidden_dim"]),
                nn.ReLU(),
                nn.Linear(model_args["moe_hidden_dim"], hidden_size)
            ) for _ in range(self.num_experts)
        ])

        self.gate = nn.Linear(hidden_size, self.num_experts)
        self.classifier = nn.Linear(hidden_size, model_args["num_classes"])

    def forward(self, **inputs):
        outputs = self.encoder(**inputs)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]

        gate_logits = self.gate(pooled)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        expert_outputs = torch.stack([expert(pooled) for expert in self.experts], dim=1)
        moe_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        return self.classifier(moe_output)
