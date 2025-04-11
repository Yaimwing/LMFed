from clients.base_client import FLClient
from models.fedmoe_model import FedMoEModel

class FedMoEClient(FLClient):
    def build_model(self, model_args):
        return FedMoEModel(model_args)
