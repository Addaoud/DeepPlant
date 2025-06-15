import torch.nn as nn
from typing import Optional, Dict
import torch
from src.seed import set_seed
from src.layers import AttentionPool
import numpy as np

# from src.utils import get_device
from src.transformer import build_transformer

set_seed()


class PredictionHead(nn.Module):
    def __init__(self, feedforward_dim, n_features):
        super(PredictionHead, self).__init__()
        self.embed_dim = feedforward_dim
        self.n_features = n_features
        self.n_genomes = len(self.n_features)
        self.linear = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(4 * self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.prediction_head = list()
        for i in range(self.n_genomes):
            self.prediction_head.append(
                nn.Sequential(
                    nn.Linear(4 * self.embed_dim, self.n_features[i]),
                    nn.Softplus(),
                )
            )
        self.prediction_head = nn.ModuleList(self.prediction_head)

    def forward(self, input: torch.Tensor, bit: int = 0):
        pred = self.linear(input)
        output = self.prediction_head[bit](pred)
        return output


class model(nn.Module):
    def __init__(self, config, transfomer, predictionHead):
        super(model, self).__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.transformer = transfomer
        self.attention_pool = AttentionPool(config.embed_dim)
        self.fc = predictionHead

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        bit: int = 0,
    ):

        if input_ids.ndim == 3:
            input_ids = input_ids.view(
                input_ids.shape[0] * input_ids.shape[1], input_ids.shape[2]
            )
        inputs_embeds = self.embeddings(input_ids)
        attention_output = self.transformer(inputs_embeds)
        hs = self.attention_pool(attention_output[:, 230:270, :])
        out = self.fc(hs, bit)
        return out, hs


def build_predictionHead(args):
    return PredictionHead(feedforward_dim=args.embed_dim, n_features=args.n_features)


def build_model(
    args,
    new_model: bool,
    model_path: Optional[str] = None,
    finetune: Optional[str] = False,
):
    transformer = build_transformer(args)
    predictionHead = build_predictionHead(args)
    network = model(
        args,
        transfomer=transformer,
        predictionHead=predictionHead,
    )
    if not new_model and model_path != None:
        print("Loading model state")
        model_pretrained_dict = torch.load(model_path)
        keys_pretrained = list(model_pretrained_dict.keys())
        keys_net = list(network.state_dict())
        model_weights = network.state_dict()
        for i in range(len(keys_pretrained)):
            model_weights[keys_net[i]] = model_pretrained_dict[keys_pretrained[i]]
        network.load_state_dict(model_weights)
        print("Model loaded with pretrained weights")
    if new_model and finetune:
        print("Loading pretrained model")
        model_pretrained_dict = torch.load(
            "/s/chromatin/m/nobackup/ahmed/DeepPlant/results/results_DeepPlant_simple/084124/model_25_02_12:15:43.pt"
        )
        keys_pretrained = list(model_pretrained_dict.keys())[:-2]
        keys_net = list(network.state_dict())
        model_weights = network.state_dict()
        for i in range(len(keys_pretrained)):
            model_weights[keys_net[i]] = model_pretrained_dict[keys_pretrained[i]]
        network.load_state_dict(model_weights)
    return network
