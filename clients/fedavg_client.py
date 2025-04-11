from clients.base_client import FLClient
from models.bert_classifier import BertClassifier

class FedAvgClient(FLClient):
    def build_model(self, model_args):
        return BertClassifier(model_args)
