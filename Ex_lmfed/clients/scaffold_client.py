# clients/scaffold_client.py

from .base_client import FLClient
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class ScaffoldClient(FLClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 보정 벡터 초기화 (ci)
        self.control = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters()}

    def local_train(self, epochs=1, lr=5e-5, server_control=None):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)

                # Scaffold 보정 항 추가: (c - ci) * w
                for (name, param) in self.model.named_parameters():
                    if server_control and name in server_control:
                        correction = server_control[name] - self.control[name]
                        loss += (correction * param).sum()

                loss.backward()
                optimizer.step()

        # 최종 업데이트된 파라미터를 저장
        updated_params = self.get_parameters()
        return updated_params

    def update_control(self, global_params, updated_params, local_lr, local_epochs):
        """
        클라이언트의 control vector(ci) 업데이트
        """
        for name in self.control:
            delta = (global_params[name] - updated_params[name]) / (local_epochs * local_lr)
            self.control[name] += delta
