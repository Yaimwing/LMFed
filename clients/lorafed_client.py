from clients.base_client import FLClient
from models.lora_classifier import LoRALinearClassifier

class LoRAFedClient(FLClient):
    def build_model(self, model_args):
        return LoRALinearClassifier(model_args)
