# server.py

import copy
import torch
from moe_model import LMFedMoEModel
from clients.base_client import FLClient

class FLServer:
    def __init__(self, model_args, clients, device="cpu"):
        self.clients = clients
        self.device = device
        self.global_model = clients[0].build_model(model_args).to(device)
        self.global_params = self.get_parameters()

    def get_parameters(self):
        return copy.deepcopy(self.global_model.state_dict())

    def set_parameters(self, new_params):
        self.global_model.load_state_dict(copy.deepcopy(new_params))
        self.global_params = new_params

    def broadcast(self, classifier_keys_to_exclude=None):
        # 선택적으로 classifier 레이어 제외하고 전달
        if classifier_keys_to_exclude:
            broadcast_params = {k: v for k, v in self.global_params.items() if k not in classifier_keys_to_exclude}
        else:
            broadcast_params = self.global_params

        for client in self.clients:
            client.set_parameters(copy.deepcopy(broadcast_params))

    def aggregate(self, updates, lr, epochs):
        total_clients = len(updates)
        aggregated = copy.deepcopy(updates[0])
        for key in aggregated.keys():
            for update in updates[1:]:
                aggregated[key] += update[key]
            aggregated[key] /= total_clients
        self.set_parameters(aggregated)
