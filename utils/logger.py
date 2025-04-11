import os
import time
import psutil
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

class Logger:
    # Initialize Logger: create SummaryWriter, start timer, and prepare storage for metrics
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        # TensorBoard SummaryWriter 저장 위치 (runs/experiment_name)
        self.writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
        # 학습 시작 시간 기록 (고해상도 타이머)
        self.start_time = time.perf_counter()
        self.peak_memory = 0.0
        # CSV 저장용 메트릭 로그 초기화
        self.log_data = {"round": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    
    def log_round(self, round_idx, accuracy, precision, recall, f1):
        # Metrics를 TensorBoard에 기록 (태그, 값, 스텝)
        self.writer.add_scalar("metrics/accuracy", accuracy, round_idx)
        self.writer.add_scalar("metrics/precision", precision, round_idx)
        self.writer.add_scalar("metrics/recall", recall, round_idx)
        self.writer.add_scalar("metrics/f1", f1, round_idx)
        # CSV 로그 데이터에 저장
        self.log_data["round"].append(round_idx)
        self.log_data["accuracy"].append(accuracy)
        self.log_data["precision"].append(precision)
        self.log_data["recall"].append(recall)
        self.log_data["f1"].append(f1)
        # 현재 메모리 사용량 측정 (MB 단위)
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / (1024 ** 2)
        self.peak_memory = max(self.peak_memory, mem_usage)
        # 메모리 사용량을 TensorBoard에 기록
        self.writer.add_scalar("resources/memory_usage", mem_usage, round_idx)

    def close(self):
        # 학습 종료 시점에 총 소요 시간 및 피크 메모리 기록
        total_time = time.perf_counter() - self.start_time
        self.writer.add_scalar("resources/total_time", total_time, 0)
        self.writer.add_scalar("resources/peak_memory", self.peak_memory, 0)
        # CSV 결과 디렉토리 생성 및 메트릭 저장
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(self.log_data).to_csv(f"results/{self.experiment_name}.csv", index=False)
        # TensorBoard Writer 닫기
        self.writer.close()
