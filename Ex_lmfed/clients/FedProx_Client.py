# clients/fedprox_client.py

from .base_client import FLClient
import torch
import torch.nn as nn
import torch.optim as optim

class FedProxClient(FLClient):
    def local_train(self, epochs=1, lr=5e-5, mu=0.01):
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        global_params = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        for _ in range(epochs):
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)

                # FedProx penalty term 추가
                prox_reg = 0
                for name, param in self.model.named_parameters():
                    prox_reg += ((param - global_params[name]) ** 2).sum()
                loss += (mu / 2) * prox_reg

                loss.backward()
                optimizer.step()
