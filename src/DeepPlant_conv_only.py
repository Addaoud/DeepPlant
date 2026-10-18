import torch.nn as nn
from typing import Optional
import torch
from src.seed import set_seed

from src.ConvNet import build_ConvNet
from src.layers import AttentionPool, build_predictionHead

set_seed()


class dilated_conv_module(nn.Module):
    def __init__(self, d_moodel: Optional[int] = 2048):
        super(dilated_conv_module, self).__init__()
        self.d_moodel = d_moodel
        self.dconv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.d_moodel,
                out_channels=self.d_moodel,
                kernel_size=5,
                padding=8,
                dilation=4,
            ),
            nn.BatchNorm1d(self.d_moodel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
        )
        self.dconv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.d_moodel,
                out_channels=self.d_moodel,
                kernel_size=5,
                padding=16,
                dilation=8,
            ),
            nn.BatchNorm1d(self.d_moodel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
        )
        self.dconv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.d_moodel,
                out_channels=self.d_moodel,
                kernel_size=5,
                padding=32,
                dilation=16,
            ),
            nn.BatchNorm1d(self.d_moodel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
        )

    def forward(self, src: torch.Tensor):
        dconv_out1 = self.dconv1(src) + src
        dconv_out2 = self.dconv2(dconv_out1) + dconv_out1
        dconv_out3 = self.dconv3(dconv_out2) + dconv_out2
        return dconv_out3


def build_dilated_block(args):
    return dilated_conv_module(d_moodel=args.embed_dim)


class model(nn.Module):
    def __init__(
        self,
        backbone,
        transfomer,
        predictionHead,
        consistency_regularization: Optional[bool] = False,
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
        src = self.backbone(input, bit)
        attention_output = self.transformer(src).permute(0, 2, 1)
        hs = self.attention_pool(attention_output)
        out = self.fc(hs, bit)
        if self.consistency_regularization:
            return (out, hs)
        else:
            return out


def build_model(
    args,
    new_model: bool,
    task: str,
    model_path: Optional[str] = None,
):
    backbone = build_ConvNet(args)
    transformer = build_dilated_block(args)
    predictionHead = build_predictionHead(args, task=task)
    network = model(
        backbone=backbone,
        transfomer=transformer,
        predictionHead=predictionHead,
        consistency_regularization=args.consistency_regularization,
    )
    if model_path != None:
        print("Loading model state")
        model_pretrained_dict = torch.load(model_path)
        keys_pretrained = list(model_pretrained_dict.keys())
        keys_net = list(network.state_dict())
        model_weights = network.state_dict()
        for i in range(len(keys_pretrained[:-2])):
            model_weights[keys_net[i]] = model_pretrained_dict[keys_pretrained[i]]
        if not new_model:
            model_weights[keys_net[-2]] = model_pretrained_dict[keys_pretrained[-2]]
            model_weights[keys_net[-1]] = model_pretrained_dict[keys_pretrained[-1]]
            print("Model state loaded")

        network.load_state_dict(model_weights)
    return network
