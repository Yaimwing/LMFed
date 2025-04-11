from clients.base_client import FLClient
from models.lmfed_moe_model import LMFedMoEModel

class LMFedClient(FLClient):
    def build_model(self, model_args):
        return LMFedMoEModel(model_args)
