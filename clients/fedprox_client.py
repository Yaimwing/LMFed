from clients.base_client import FLClient
from models.bert_classifier import BertClassifier
import torch

class FedProxClient(FLClient):
    def build_model(self, model_args):
        return BertClassifier(model_args)

    def local_train(self, epochs=1, lr=5e-5, mu=0.01):
        self.model.train()
        global_weights = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

        for _ in range(epochs):
            for batch in self.dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = self.criterion(logits, labels)

                # FedProx proximal term
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    proximal_term += ((param - global_weights[name]) ** 2).sum()
                loss += (mu / 2) * proximal_term

                loss.backward()
                self.optimizer.step()