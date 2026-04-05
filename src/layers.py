import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from typing import Optional
import math


class Conv1dWithRC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias: Optional[bool] = True,
        complement: Optional[bool] = False,
        **kwargs
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, bias=bias, **kwargs
        )
        self.complement = complement

    def reverse_complement_filters(self, filters):
        rc = torch.flip(filters, dims=[-1])  # reverse along sequence axis
        swap = torch.tensor([3, 2, 1, 0], device=filters.device)  # index mapping
        rc = rc[:, swap, :]
        return rc

    def reverse_filters(self, filters):
        r = torch.flip(filters, dims=[-1])  # reverse along sequence axis
        return r

    def forward(self, x):
        # x: (batch, 4, length)
        w = self.conv.weight  # (out_channels, 4, kernel_size)
        b = self.conv.bias

        # compute RC filters

        if self.complement:
            w_rc = self.reverse_complement_filters(w)
            return F.conv1d(
                x,
                w,
                b,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            ) + F.conv1d(
                x,
                w_rc,
                b,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            )

        else:
            w_r = self.reverse_filters(w)
            return F.conv1d(
                x,
                w,
                b,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            ) + F.conv1d(
                x,
                w_r,
                b,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
            )


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super(AttentionPool, self).__init__()
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x: torch.Tensor):
        attn_logits = torch.einsum("b n d, d e -> b n e", x, self.to_attn_logits)
        attn = attn_logits.softmax(dim=-2)
        return (x * attn).sum(dim=-2).squeeze(dim=-2)


class CSPHead(nn.Module):
    def __init__(self, embed_dim, expand_factor, n_features):
        super(CSPHead, self).__init__()
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.n_genomes = len(self.n_features)
        self.expand_factor = expand_factor
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
                    nn.Linear(self.expand_factor * self.embed_dim, self.n_features[i]),
                    nn.Softplus(beta=1.0, threshold=10.0),
                )
            )
        self.prediction_head = nn.ModuleList(self.prediction_head)

    def forward(self, input: torch.Tensor, bit: int = 0):
        pred = self.linear(input)
        output = self.prediction_head[bit](pred)
        return output


class GEPHead(nn.Module):
    def __init__(self, embed_dim, expand_factor, n_features):
        super(GEPHead, self).__init__()
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.n_genomes = len(self.n_features)
        self.expand_factor = expand_factor
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
                    nn.Linear(self.expand_factor * self.embed_dim, self.n_features[i]),
                    nn.Softplus(beta=1.0, threshold=10.0),
                )
            )
        self.prediction_head = nn.ModuleList(self.prediction_head)

    def forward(self, input: torch.Tensor, bit: int = 0):
        pred = self.linear(input)
        output = self.prediction_head[bit](pred)
        return output


class EAPHead(nn.Module):
    def __init__(self, embed_dim, expand_factor, n_features):
        super(EAPHead, self).__init__()
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.n_genomes = len(self.n_features)
        self.expand_factor = expand_factor
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
                    nn.Linear(self.expand_factor * self.embed_dim, self.n_features[i]),
                )
            )
        self.prediction_head = nn.ModuleList(self.prediction_head)

    def forward(self, input: torch.Tensor, bit: int = 0):
        pred = self.linear(input)
        output = self.prediction_head[bit](pred)
        return output


def build_predictionHead(args, task="CSP"):
    if task == "CSP":
        return CSPHead(
            embed_dim=args.embed_dim,
            expand_factor=args.expand_factor,
            n_features=args.n_features,
        )
    elif task == "GEP":
        return GEPHead(
            embed_dim=args.embed_dim,
            expand_factor=args.expand_factor,
            n_features=args.n_features,
        )
    elif task == "EAP":
        return EAPHead(
            embed_dim=args.embed_dim,
            expand_factor=args.expand_factor,
            n_features=args.n_features,
        )
