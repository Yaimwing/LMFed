import copy
from servers.fl_server import FLServer
import torch

class ScaffoldServer(FLServer):
    def __init__(self, model_args, clients, device="cpu"):
        super().__init__(model_args, clients, device)
        self.control = {k: torch.zeros_like(v) for k, v in self.global_model.state_dict().items()}

    def broadcast(self, classifier_keys_to_exclude=None):
        super().broadcast(classifier_keys_to_exclude)
        for client in self.clients:
            client.set_parameters(copy.deepcopy(self.global_params))

    def aggregate(self, updates, lr, epochs):
        total_clients = len(updates)
        new_params = copy.deepcopy(updates[0])

        for key in new_params:
            for update in updates[1:]:
                new_params[key] += update[key]
            new_params[key] /= total_clients

        self.update_control_variates(new_params, lr, epochs)
        self.set_parameters(new_params)

    def update_control_variates(self, new_params, lr, epochs):
        for key in self.control:
            self.control[key] += (self.global_params[key] - new_params[key]) / (lr * epochs)