import copy
import torch

class FLServer:
    def __init__(self, model_args, clients, device="cpu"):
        self.model_args = model_args
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
        self.classifier_keys_to_exclude = classifier_keys_to_exclude or []
        broadcast_params = copy.deepcopy(self.global_params)
        if classifier_keys_to_exclude:
            for key in classifier_keys_to_exclude:
                if key in broadcast_params:
                    del broadcast_params[key]
        for client in self.clients:
            client.set_parameters(copy.deepcopy(broadcast_params))

    def aggregate(self, updates, lr, epochs):
        total_clients = len(updates)
        aggregated = copy.deepcopy(updates[0])

        for key in aggregated:
            if hasattr(self, "classifier_keys_to_exclude") and key in self.classifier_keys_to_exclude:
                continue

            try:
                aggregated[key] = aggregated[key].float()
                for update in updates[1:]:
                    aggregated[key] += update[key].float()
                aggregated[key] = aggregated[key] / total_clients
            except RuntimeError as e:
                print(f"[Warning] Skipping aggregation for key: {key} due to shape mismatch ({e})")

        self.set_parameters(aggregated)