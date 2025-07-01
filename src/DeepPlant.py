import torch.nn as nn
from typing import Optional
import torch
from src.seed import set_seed

from src.transformer import build_transformer
from src.ConvNet import build_ConvNet
from src.layers import AttentionPool, build_predictionHead

set_seed()


class model(nn.Module):
    def __init__(
        self,
        backbone,
        transfomer,
        predictionHead,
        consistency_regularization: Optional[bool] = True,
    ):
        super(model, self).__init__()
        self.backbone = backbone
        self.transformer = transfomer
        num_channels = backbone.num_channels
        embed_dim = predictionHead.embed_dim
        self.attention_pool = AttentionPool(embed_dim)
        self.consistency_regularization = consistency_regularization

        self.fc = predictionHead

    def get_convlayers(self, bit: int = 0):
        for i in range(self.backbone.conv_features.__len__()):
            yield (
                self.backbone.conv_features[i][0].weight
                if i == bit
                else self.backbone.conv_features[i][0].weight.detach()
            )

    def forward(self, input: torch.Tensor, bit: int = 0):
        if input.ndim == 4:
            input = input.view(
                input.shape[0] * input.shape[1], input.shape[2], input.shape[3]
            )
        src = self.backbone(input, bit).permute(0, 2, 1)
        attention_output = self.transformer(src)
        hs = self.attention_pool(attention_output)
        out = self.fc(hs, bit)
        if self.consistency_regularization:
            return (out, hs)
        else:
            return out


def build_model(
    args,
    new_model: bool,
    model_path: Optional[str] = None,
    finetune: Optional[str] = False,
):
    backbone = build_ConvNet(args)
    transformer = build_transformer(args)
    predictionHead = build_predictionHead(args)
    network = model(
        backbone=backbone,
        transfomer=transformer,
        predictionHead=predictionHead,
        consistency_regularization=args.consistency_regularization,
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
