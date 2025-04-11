from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import numpy as np

class MetricCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.labels = []

    def update(self, logits, labels):
        if isinstance(logits, torch.Tensor):
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            labels = labels.cpu().tolist()
        else:
            preds = logits

        self.predictions.extend(preds)
        self.labels.extend(labels)

    def compute(self):
        accuracy = accuracy_score(self.labels, self.predictions)
        f1 = f1_score(self.labels, self.predictions, average="weighted")
        f1_macro = f1_score(self.labels, self.predictions, average="macro")
        f1_micro = f1_score(self.labels, self.predictions, average="micro")
        precision = precision_score(self.labels, self.predictions, average="weighted", zero_division=0)
        recall = recall_score(self.labels, self.predictions, average="weighted", zero_division=0)
        conf_matrix = confusion_matrix(self.labels, self.predictions)

        num_classes = max(max(self.labels), max(self.predictions)) + 1
        per_class_acc = {}
        for i in range(num_classes):
            true_i = [l == i for l in self.labels]
            pred_i = [p == i for p in self.predictions]
            correct_i = sum([t and p for t, p in zip(true_i, pred_i)])
            total_i = sum(true_i)
            per_class_acc[f"acc_class_{i}"] = correct_i / total_i if total_i > 0 else 0.0

        return {
            "accuracy": accuracy,
            "f1": f1,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "precision": precision,
            "recall": recall,
            **per_class_acc,
            "confusion_matrix": conf_matrix.tolist()  # for JSON serialization
        }
