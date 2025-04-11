import os
import time
import torch
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset_manager import FederatedDatasetManager
from clients.fedavg_client import FedAvgClient
from clients.LoRAFed_Client import LoRAFedClient
from clients.fedMoE_client import FedMoEClient
from clients.FedProx_Client import FedProxClient
from clients.scaffold_client import ScaffoldClient
from clients.lmfed_client import LMFedClient

from scaffold_server import ScaffoldServer
from server import FLServer

def run_experiment(client_class, server_class, model_args, label, rounds=10, local_epochs=1, local_lr=5e-5):
    print(f"\n🚀 Running experiment: {label}")

    writer = SummaryWriter(log_dir=f"runs/{label}")
    manager = FederatedDatasetManager()
    client_datasets = manager.prepare_all_clients()

    clients = []
    for cid, ds in client_datasets.items():
        dynamic_args = model_args.copy()
        dynamic_args["num_classes"] = getattr(ds, "num_classes", 2)

        clients.append(
            client_class(
                client_id=cid,
                dataset=ds,
                model_args=dynamic_args,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        )

    # ✅ 서버 모델도 첫 클라이언트 기준으로 num_classes 통일
    first_client_dataset = list(client_datasets.values())[0]
    global_model_args = model_args.copy()
    global_model_args["num_classes"] = getattr(first_client_dataset, "num_classes", 2)

    server = server_class(
        model_args=global_model_args,
        clients=clients,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    process = psutil.Process(os.getpid())

    log = {"round": [], "global_acc": []}
    detailed_log = {"round": [], "client": [], "local_time": [], "memory_mb": [], "param_sent": []}
    for client in clients:
        log[f"{client.client_id}_acc"] = []

    for rnd in tqdm(range(1, rounds + 1), desc="Rounds"):
        server.broadcast(classifier_keys_to_exclude=["classifier.weight", "classifier.bias"])

        updated_list = []
        for client in tqdm(clients, desc=f"Round {rnd} Clients"):
            start_time = time.perf_counter()

            if isinstance(server, ScaffoldServer):
                updated = client.local_train(epochs=local_epochs, lr=local_lr, server_control=server.control)
                client.update_control(server.global_params, updated, local_lr, local_epochs)
            else:
                client.local_train(epochs=local_epochs, lr=local_lr)
                updated = client.get_parameters()

            elapsed_time = time.perf_counter() - start_time
            updated_list.append(updated)

            if hasattr(client, "get_differential_updates"):
                diff = client.get_differential_updates(server.global_params)
                param_sent = sum(p.numel() for p in diff.values())
            else:
                param_sent = sum(p.numel() for p in updated.values())

            mem_usage = process.memory_info().rss / (1024 ** 2)

            detailed_log["round"].append(rnd)
            detailed_log["client"].append(client.client_id)
            detailed_log["local_time"].append(elapsed_time)
            detailed_log["memory_mb"].append(mem_usage)
            detailed_log["param_sent"].append(param_sent)

            writer.add_scalar(f"client/{client.client_id}/time", elapsed_time, rnd)
            writer.add_scalar(f"client/{client.client_id}/memory", mem_usage, rnd)
            writer.add_scalar(f"client/{client.client_id}/params_sent", param_sent, rnd)

        server.aggregate(updated_list, local_lr, local_epochs)

        round_acc = 0
        for client in clients:
            acc = client.evaluate_local()
            log[f"{client.client_id}_acc"].append(acc)
            writer.add_scalar(f"client/{client.client_id}/accuracy", acc, rnd)
            round_acc += acc

        round_acc /= len(clients)
        log["round"].append(rnd)
        log["global_acc"].append(round_acc)
        writer.add_scalar("global/accuracy", round_acc, rnd)
        print(f"  → Round {rnd} Global Accuracy: {round_acc:.2%}")

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(log)
    df.to_csv(f"results/{label}.csv", index=False)

    df_detail = pd.DataFrame(detailed_log)
    df_detail.to_csv(f"results/{label}_detailed.csv", index=False)

    plt.figure()
    for client in clients:
        plt.plot(df["round"], df[f"{client.client_id}_acc"], label=client.client_id)
    plt.plot(df["round"], df["global_acc"], label="Global", linestyle="--", linewidth=2)
    plt.title(f"Accuracy over Rounds - {label}")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"results/{label}_plot.png")
    print(f"✔ Results saved: results/{label}.csv, {label}_detailed.csv, {label}_plot.png")
    writer.close()

if __name__ == "__main__":
    model_config = {
        "model_type": "bert",
        "model_name": "bert-base-uncased",
        "num_experts": 4,
        "moe_hidden_dim": 512,
        "lora_rank": 4
        # num_classes는 각 클라이언트 및 서버에서 자동 설정됨
    }

    experiments = [
        ("LMFed", LMFedClient, FLServer),
        ("FedAvg", FedAvgClient, FLServer),
        ("LoRAFed", LoRAFedClient, FLServer),
        ("FedMoE", FedMoEClient, FLServer),
        ("FedProx", FedProxClient, FLServer),
        ("Scaffold", ScaffoldClient, ScaffoldServer),
    ]

    for label, client_cls, server_cls in experiments:
        run_experiment(
            client_class=client_cls,
            server_class=server_cls,
            model_args=model_config,
            label=label,
            rounds=10,
            local_epochs=1
        )
