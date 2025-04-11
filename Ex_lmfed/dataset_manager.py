# dataset_manager.py

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.num_classes = len(set(labels))  # ✅ 클래스 수 저장

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class FederatedDatasetManager:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_dataset(self, dataset, text_column, label_column):
        encodings = self.tokenizer(list(dataset[text_column]), truncation=True, padding=True, max_length=128)
        labels = list(dataset[label_column])
        return CustomTextDataset(encodings, labels)

    def prepare_all_clients(self):
        datasets = {}

        task_configs = [
            ("sst2", "sentence", "label"),
            ("mnli", "premise", "label"),
            ("qqp", "question1", "label"),
            ("ag_news", "text", "label"),
            ("imdb", "text", "label")
        ]

        hf_datasets = [
            load_dataset("glue", "sst2")["train"].shuffle(seed=42).select(range(500)),
            load_dataset("glue", "mnli")["train"].shuffle(seed=42).select(range(500)),
            load_dataset("glue", "qqp")["train"].shuffle(seed=42).select(range(500)),
            load_dataset("ag_news")["train"].shuffle(seed=42).select(range(500)),
            load_dataset("imdb")["train"].shuffle(seed=42).select(range(500))
        ]

        for i, ((name, text_col, label_col), dataset) in enumerate(zip(task_configs, hf_datasets)):
            ds = self.tokenize_dataset(dataset, text_col, label_col)
            client_id = f"C{i+1}_{name}"
            datasets[client_id] = ds

        return datasets
