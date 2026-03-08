import torch.nn as nn
from typing import Optional
import torch
from src.seed import set_seed
import numpy as np

# from src.utils import get_device
from src.transformer import build_transformer
from src.ConvNetRC1 import build_ConvNet
from src.layers import AttentionPool

set_seed()


class PredictionHead(nn.Module):
    def __init__(self, embed_dim, expand_factor, n_features, targets):
        super(PredictionHead, self).__init__()
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.n_genomes = len(self.n_features)
        self.expand_factor = expand_factor
        self.n_targets = 2835
        self.targets = targets  # 'ALL' 'HM'
        if self.targets == "TF":
            self.ind = np.load(
                "/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/TF_idx.npy"
            )
            self.n_targets = self.ind.shape[0]
            self.means = torch.from_numpy(
                np.load("/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/TF_mean.npy")
            ).float()
            self.stds = torch.from_numpy(
                np.load("/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/TF_std.npy")
            ).float()
        elif self.targets == "HM":
            self.ind = np.load(
                "/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/HM_idx.npy"
            )
            self.n_targets = self.ind.shape[0]
            self.means = torch.from_numpy(
                np.load("/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/HM_mean.npy")
            ).float()
            self.stds = torch.from_numpy(
                np.load("/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/HM_std.npy")
            ).float()
        elif self.targets == "TF_withTET1":
            self.ind = np.load(
                "/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/TF_idx_withTET1.npy"
            )
            self.n_targets = self.ind.shape[0]
            self.means = torch.from_numpy(
                np.load(
                    "/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/TF_withTET1_mean.npy"
                )
            ).float()
            self.stds = torch.from_numpy(
                np.load(
                    "/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/TF_withTET1_std.npy"
                )
            ).float()
        else:
            self.means = torch.from_numpy(
                np.load("/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/All_mean.npy")
            ).float()
            self.stds = torch.from_numpy(
                np.load("/s/chromatin/m/nobackup/ahmed/DeepPlant/Metadata/All_std.npy")
            ).float()

        self.linear = nn.Sequential(
            nn.Linear(self.embed_dim, self.expand_factor * self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(
                self.expand_factor * self.embed_dim, self.expand_factor * self.embed_dim
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.prediction_head = list()
        for i in range(self.n_genomes):
            self.prediction_head.append(
                nn.Sequential(
                    nn.Linear(self.expand_factor * self.embed_dim, 2835),
                    nn.Softplus(beta=1.0, threshold=10.0),
                )
            )
        self.prediction_head = nn.ModuleList(self.prediction_head)
        self.final_layer = nn.Sequential(
            nn.Linear(self.n_targets, self.n_features[i], bias=False),
        )

    def forward(self, input: torch.Tensor, bit: int = 0):
        pred = self.linear(input)
        output = self.prediction_head[bit](pred)
        if self.targets != "ALL":
            output = output[:, self.ind]
        output = output / self.means.to(output.device)  # / self.stds.to(output.device)
        output = self.final_layer(output)
        return output.squeeze(1)


class model(nn.Module):
    def __init__(self, backbone, transfomer, predictionHead):
        super(model, self).__init__()
        self.backbone = backbone
        self.transformer = transfomer
        num_channels = backbone.num_channels
        embed_dim = predictionHead.embed_dim
        self.attention_pool = AttentionPool(embed_dim)

        self.fc = predictionHead

    def train(self, mode: bool = True):
        # Call the base implementation first
        super().train(mode)

        # Force frozen modules to stay in eval mode
        if mode:
            self.backbone.eval()
            self.transformer.eval()
            self.attention_pool.eval()
            self.fc.linear.eval()
            self.fc.prediction_head.eval()
        return self

    def forward(self, input: torch.Tensor, bit: int = 0):
        if input.ndim == 4:
            input = input.view(
                input.shape[0] * input.shape[1], input.shape[2], input.shape[3]
            )
        src = self.backbone(input, bit).permute(0, 2, 1)
        attention_output = self.transformer(src)
        hs = self.attention_pool(attention_output)
        out = self.fc(hs, bit)
        return out


def build_predictionHead(args, targets):
    return PredictionHead(
        embed_dim=args.embed_dim,
        expand_factor=args.expand_factor,
        n_features=args.n_features,
        targets=targets,
    )


def build_model(
    args,
    new_model: bool,
    model_path: Optional[str] = None,
    targets: Optional[str] = "HM",
):
    backbone = build_ConvNet(args)
    transformer = build_transformer(args)
    predictionHead = build_predictionHead(args, targets)
    network = model(
        backbone=backbone, transfomer=transformer, predictionHead=predictionHead
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
    if new_model:
        print("Loading pretrained model state")
        model_pretrained_dict = torch.load(
            "/s/chromatin/a/nobackup/ahmed/DeepPlant/results/DeepPlant_AT/177318/model_25_12_18:05:28_new.pt"
        )
        keys_pretrained = list(model_pretrained_dict.keys())
        keys_net = list(network.state_dict())
        model_weights = network.state_dict()
        for i in range(len(keys_pretrained)):
            model_weights[keys_net[i]] = model_pretrained_dict[keys_pretrained[i]]
            # model_weights[keys_net[i]].requires_grad = False
        network.load_state_dict(model_weights)
    for name, param in network.named_parameters():
        if "final_layer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return network
