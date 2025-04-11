# clients/fedmoe_client.py

from .base_client import FLClient
import torch
import torch.nn as nn
import torch.optim as optim

class FedMoEClient(FLClient):
    def local_train(self, epochs=1, lr=5e-5):
        self.model.train()
        # Sub-MoE 전체 파라미터를 학습 (LoRA 없음)
        moe_params = [p for n, p in self.model.named_parameters() if "moe" in n]
        optimizer = optim.AdamW(moe_params, lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
