import copy

class BaseServer:
    def __init__(self, model, clients, device="cpu"):
        self.model = model.to(device)
        self.clients = clients
        self.device = device
        self.global_params = copy.deepcopy(self.model.state_dict())

    def broadcast(self, keys_to_exclude=None):
        if keys_to_exclude:
            broadcast_params = {
                k: v for k, v in self.global_params.items()
                if k not in keys_to_exclude
            }
        else:
            broadcast_params = self.global_params

        for client in self.clients:
            client.set_parameters(copy.deepcopy(broadcast_params))

    def aggregate(self, client_updates):
        total = len(client_updates)
        agg = copy.deepcopy(client_updates[0])
        for key in agg:
            for update in client_updates[1:]:
                agg[key] += update[key]
            agg[key] /= total
        self.set_parameters(agg)

    def set_parameters(self, new_params):
        self.model.load_state_dict(copy.deepcopy(new_params))
        self.global_params = new_params

    def get_parameters(self):
        return copy.deepcopy(self.model.state_dict())