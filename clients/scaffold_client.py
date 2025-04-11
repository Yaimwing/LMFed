from clients.base_client import FLClient
import copy
import torch
from models.lmfed_moe_model import LMFedMoEModel

class ScaffoldClient(FLClient):
    def __init__(self, client_id, dataset, model_args, device="cpu"):
        super().__init__(client_id, dataset, model_args, device)
        self.control = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters()}

    def build_model(self, model_args):
        return LMFedMoEModel(model_args)

    def local_train(self, epochs=1, lr=5e-5, server_control=None):
        self.model.train()
        for _ in range(epochs):
            for batch in self.dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = self.criterion(logits, labels)
                loss.backward()

                # ✅ Scaffold correction with shape check
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in self.control and name in server_control:
                        if self.control[name].shape == server_control[name].shape:
                            param.grad += self.control[name] - server_control[name]

                self.optimizer.step()

    def get_control_update(self, server_control):
        control_update = {}
        for name, param in self.model.named_parameters():
            if name in server_control and param.requires_grad:
                if self.control[name].shape == server_control[name].shape:
                    control_update[name] = self.control[name] - server_control[name]
        return control_update
