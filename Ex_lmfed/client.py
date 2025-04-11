# client.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from moe_model import LMFedMoEModel

class FLClient:
    def __init__(self, 
                 client_id: str,
                 dataset,
                 model_args: dict,
                 batch_size: int = 16,
                 device: str = "cpu"):
        self.client_id = client_id
        self.device = device
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 모델 초기화 (client도 로컬에 full 모델 구조 가짐)
        self.model = LMFedMoEModel(**model_args).to(self.device)

    def set_parameters(self, global_params: dict):
        """
        서버로부터 받은 파라미터로 로컬 모델 동기화
        """
        self.model.load_state_dict(copy.deepcopy(global_params))

    def get_parameters(self) -> dict:
        """
        로컬 모델의 전체 state_dict 반환
        """
        return copy.deepcopy(self.model.state_dict())

    def local_train(self, epochs=1, lr=5e-5):
        """
        로컬 데이터셋으로 모델 학습 (LoRA 파라미터만 학습)
        """
        self.model.train()

        # LoRA 파라미터만 업데이트
        lora_params = [p for n, p in self.model.named_parameters() if "A" in n or "B" in n]
        optimizer = optim.AdamW(lora_params, lr=lr)
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

    def get_differential_updates(self, global_params: dict, threshold=1e-4) -> dict:
        """
        글로벌 파라미터와 로컬 파라미터 간의 차이 중, threshold 이상인 것만 반환
        """
        local_params = self.model.state_dict()
        diff = {}

        for key in local_params:
            if key in global_params:
                delta = local_params[key] - global_params[key]
                if torch.abs(delta).max().item() > threshold:
                    diff[key] = delta.clone().detach()
        return diff

    def evaluate(self, dataloader=None):
        """
        로컬 모델로 간단 평가 (정확도 측정)
        """
        self.model.eval()
        correct = 0
        total = 0
        loader = dataloader if dataloader else self.dataloader
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0
