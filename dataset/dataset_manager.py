from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

class CustomTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.num_classes = len(set(labels))

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class FederatedDatasetManager:
    def __init__(self, sample_size=200):  # ✅ 클라이언트당 샘플 수 지정 가능
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.sample_size = sample_size

    def tokenize_dataset(self, dataset, text_column, label_column):
        encodings = self.tokenizer(list(dataset[text_column]), truncation=True, padding=True, max_length=128)
        labels = list(dataset[label_column])
        return CustomTextDataset(encodings, labels)

    def prepare_all_clients(self):
        datasets = {}
        task_configs = [
            ("C1_SST2", "glue", "sst2", "sentence", "label"),
            ("C2_MNLI", "glue", "mnli", "premise", "label"),
            ("C3_QQP", "glue", "qqp", "question1", "label"),
            ("C4_AGNews", "ag_news", None, "text", "label"),
            ("C5_IMDB", "imdb", None, "text", "label")
        ]

        for cid, dataset_name, subset, text_col, label_col in task_configs:
            if subset:
                ds_raw = load_dataset(dataset_name, subset)["train"].shuffle(seed=42).select(range(self.sample_size))
            else:
                ds_raw = load_dataset(dataset_name)["train"].shuffle(seed=42).select(range(self.sample_size))

            ds = self.tokenize_dataset(ds_raw, text_col, label_col)
            datasets[cid] = ds

        return datasets