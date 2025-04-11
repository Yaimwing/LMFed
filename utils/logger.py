import os
import time
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

class ExperimentLogger:
    def __init__(self, log_dir, experiment_name):
        self.start_time = time.time()
        self.log_path = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_path)

    def log_metrics(self, round_num, accuracy, memory_mb=None, communication_kb=None, label=""):
        prefix = f"{label}/" if label else ""
        self.writer.add_scalar(f"{prefix}accuracy", accuracy, round_num)
        if memory_mb is not None:
            self.writer.add_scalar(f"{prefix}memory_MB", memory_mb, round_num)
        if communication_kb is not None:
            self.writer.add_scalar(f"{prefix}communication_KB", communication_kb, round_num)

    def log_time(self, round_num, label=""):
        elapsed = time.time() - self.start_time
        prefix = f"{label}/" if label else ""
        self.writer.add_scalar(f"{prefix}elapsed_time_sec", elapsed, round_num)

    def get_current_memory(self):
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # in MB

    def close(self):
        self.writer.close()
