import os
import time
import torch
import yaml
import psutil
from tqdm import tqdm
import importlib
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset_manager import FederatedDatasetManager
from utils.logger import Logger
from utils.metric import MetricCalculator


def load_config(path="config/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_class_from_name(class_name):
    base = class_name.replace("Client", "").lower()
    module_path = f"clients.{base}_client"
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_server_class(class_name):
    base = class_name.replace("Server", "").lower()
    module_path = f"servers.{base}_server"
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def run_experiment(client_class, server_class, model_args, label, rounds, local_epochs, lr):
    writer = SummaryWriter(log_dir=f"runs/{label}")
    logger = Logger(experiment_name=label)
    manager = FederatedDatasetManager()
    client_datasets = manager.prepare_all_clients()

    clients = []
    for cid, ds in client_datasets.items():
        args = model_args.copy()
        args["num_classes"] = getattr(ds, "num_classes", 2)
        client = client_class(cid, ds, args, device="cuda" if torch.cuda.is_available() else "cpu")
        clients.append(client)

    global_args = model_args.copy()
    global_args["num_classes"] = clients[0].model_args["num_classes"]
    server = server_class(model_args=global_args, clients=clients, device="cuda" if torch.cuda.is_available() else "cpu")

    for rnd in tqdm(range(1, rounds + 1), desc=f"{label}"):
        server.broadcast(classifier_keys_to_exclude=["classifier.weight", "classifier.bias"])
        updates = []

        for client in clients:
            if label == "Scaffold":
                client.local_train(epochs=local_epochs, lr=lr, server_control=server.control)
            elif label == "FedProx":
                client.local_train(epochs=local_epochs, lr=lr, mu=0.01)
            else:
                client.local_train(epochs=local_epochs, lr=lr)
            updates.append(client.get_parameters())

        server.aggregate(updates, lr, local_epochs)

        metrics = MetricCalculator()
        for client in clients:
            client.model.eval()
            for batch in client.dataloader:
                inputs = {k: v.to(client.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(client.device)
                with torch.no_grad():
                    logits = client.model(**inputs)
                metrics.update(logits, labels)
        results = metrics.compute()
        logger.log_round(rnd, results["accuracy"], results["precision"], results["recall"], results["f1"])
        writer.add_scalar("global/accuracy", results["accuracy"], rnd)
        print(f"[Round {rnd}] {label} Accuracy: {results['accuracy']:.4f}")

    logger.close()
    writer.close()

if __name__ == "__main__":
    config = load_config()
    model_args = config["model"]
    rounds = config["training"]["rounds"]
    local_epochs = config["training"]["local_epochs"]
    lr = float(model_args.get("lr", 5e-5))  # ✅ 문자열이면 float으로 강제 변환

    for client_conf in config["clients"]:
        name = client_conf["name"]
        client_class = get_class_from_name(client_conf["class"])
        if name == "Scaffold":
            server_class = get_server_class(config["server"]["scaffold_class"])
        else:
            server_class = get_server_class(config["server"]["default_class"])

        run_experiment(client_class, server_class, model_args, label=name, rounds=rounds, local_epochs=local_epochs, lr=lr)
