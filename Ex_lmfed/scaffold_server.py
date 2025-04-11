# servers/scaffold_server.py

import torch
import copy
from moe_model import LMFedMoEModel

class ScaffoldServer:
    def __init__(self, model_args, clients, device="cpu"):
        self.device = device
        self.clients = clients
        self.global_model = LMFedMoEModel(**model_args).to(device)
        self.global_params = self.global_model.state_dict()

        # 서버 보정 벡터 (control variate)
        self.control = {name: torch.zeros_like(param.data) for name, param in self.global_model.named_parameters()}

    def broadcast(self):
        for client in self.clients:
            client.set_parameters(copy.deepcopy(self.global_params))

    def aggregate(self, updated_params_list, local_lr, local_epochs):
        n = len(updated_params_list)
        new_params = copy.deepcopy(self.global_params)

        for name in new_params:
            delta_sum = sum([updated[name] - self.global_params[name] for updated in updated_params_list])
            avg_delta = delta_sum / n
            new_params[name] = self.global_params[name] + avg_delta

            # 서버 보정 벡터 업데이트
            self.control[name] += avg_delta / (local_lr * local_epochs)

        self.global_params = new_params
        self.global_model.load_state_dict(self.global_params)

    def evaluate_global(self, test_dataloader):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits, _ = self.global_model(input_ids=input_ids, attention_mask=attention_mask)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0
