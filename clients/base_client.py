import copy
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class FLClient:
    def __init__(self, client_id, dataset, model_args, device="cpu"):
        self.client_id = client_id
        self.dataset = dataset
        self.model_args = model_args
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_args.get("model_name", "bert-base-uncased"))
        self.model = self.build_model(model_args).to(device)

        self.criterion = CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=float(model_args.get("lr", 5e-5)))  # ✅ float 변환 추가
        self.dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    def build_model(self, model_args):
        raise NotImplementedError("build_model must be implemented by subclass")

    def get_parameters(self):
        return copy.deepcopy(self.model.state_dict())

    def set_parameters(self, global_params):
        current_state = self.model.state_dict()
        for k, v in global_params.items():
            if not k.startswith("classifier."):
                current_state[k] = v
        self.model.load_state_dict(current_state)

    def local_train(self, epochs=1, lr=5e-5):
        self.model.train()
        print(f"[{self.client_id}] training start")
        for _ in range(epochs):
            for batch in self.dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate_local(self):
        self.model.eval()
        correct = 0
        total = 0
        eval_loader = DataLoader(self.dataset, batch_size=32)

        with torch.no_grad():
            for batch in eval_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0
