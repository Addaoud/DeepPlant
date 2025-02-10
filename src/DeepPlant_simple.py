import torch.nn as nn
from typing import Optional, Dict
import torch
from src.seed import set_seed
from scipy.interpolate import splev
import numpy as np

# from src.utils import get_device
from src.transformer import build_transformer
from src.ConvNet import build_ConvNet

set_seed()


def bs(x, df=None, knots=None, degree=3, intercept=False):
    """
    df : int
        The number of degrees of freedom to use for this spline. The
        return value will have this many columns. You must specify at least
        one of `df` and `knots`.
    knots : list(float)
        The interior knots of the spline. If unspecified, then equally
        spaced quantiles of the input data are used. You must specify at least
        one of `df` and `knots`.
    degree : int
        The degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool
        If `True`, the resulting spline basis will span the intercept term
        (i.e. the constant function). If `False` (the default) then this
        will not be the case, which is useful for avoiding overspecification
        in models that include multiple spline terms and/or an intercept term.
    """
    order = degree + 1
    inner_knots = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            print("df was too small; have used %d" % (order - (1 - intercept)))
        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1]
            )
    elif knots is not None:
        inner_knots = knots
    all_knots = np.concatenate(([np.min(x), np.max(x)] * order, inner_knots))
    all_knots.sort()
    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)
    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))
    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n, df, log=False):
    if log:
        dist = np.array(np.arange(n) - n / 2.0)
        dist = np.log(np.abs(dist) + 1) * (2 * (dist > 0) - 1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist), np.max(dist), n_knots + 2)[1:-1]
        return torch.from_numpy(bs(dist, knots=knots, intercept=True)).float()
    else:
        dist = np.arange(n)
        return torch.from_numpy(bs(dist, df=df, intercept=True)).float()


class BSplineTransformation(nn.Module):
    def __init__(self, bins, log=False, scaled=False):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = bins

    def forward(self, input: torch.Tensor):
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            if input.is_cuda:
                self._spline_tr = self._spline_tr.cuda()
        return torch.matmul(input, self._spline_tr)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super(AttentionPool, self).__init__()
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x: torch.Tensor):
        attn_logits = torch.einsum("b n d, d e -> b n e", x, self.to_attn_logits)
        attn = attn_logits.softmax(dim=-2)
        return (x * attn).sum(dim=-2).squeeze(dim=-2)


class PredictionHead(nn.Module):
    def __init__(self, feedforward_dim, n_features):
        super(PredictionHead, self).__init__()
        self.embed_dim = feedforward_dim
        self.n_features = n_features
        self.n_genomes = len(self.n_features)
        self.spline_bins = 16
        self.spline_tr = BSplineTransformation(bins=self.spline_bins, scaled=False)
        # self.attention_pool = AttentionPool(self.embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(16 * self.embed_dim, 16 * self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(16 * self.embed_dim, 12 * self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(12 * self.embed_dim, 8 * self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.prediction_head = list()
        for i in range(self.n_genomes):
            self.prediction_head.append(
                nn.Sequential(
                    nn.Linear(8 * self.embed_dim, self.n_features[i]),
                    nn.Softplus(),
                )
            )
        self.prediction_head = nn.ModuleList(self.prediction_head)

    def forward(self, input: torch.Tensor, bit: int = 0):
        # reshape_out = input.mean(dim=-2).squeeze(dim=-2)
        # reshape_out = self.attention_pool(input)
        spline_out = self.spline_tr(input)
        reshape_out = spline_out.view(
            spline_out.size(0), self.spline_bins * self.embed_dim
        )
        pred = self.linear(reshape_out)
        output = self.prediction_head[bit](pred)
        return output


class model(nn.Module):
    def __init__(self, backbone, transfomer, predictionHead):
        super(model, self).__init__()
        self.backbone = backbone
        self.transformer = transfomer
        num_channels = backbone.num_channels
        embed_dim = predictionHead.embed_dim
        self.input_proj = nn.Conv1d(num_channels, embed_dim, kernel_size=1)
        # self.query_embed = nn.Embedding(num_class, embed_dim)
        self.fc = predictionHead

    def get_convlayers(self, bit: int = 0):
        for i in range(self.backbone.conv_features.__len__()):
            yield (
                self.backbone.conv_features[i][0].weight
                if i == bit
                else self.backbone.conv_features[i][0].weight.detach()
            )

    def forward(self, input: torch.Tensor, bit: int = 0):
        src = self.backbone(input, bit)
        src = self.input_proj(src)
        src = src.permute(0, 2, 1)
        hs = self.transformer(src).transpose(1, 2)
        out = self.fc(hs, bit)
        return out


def build_predictionHead(args):
    return PredictionHead(feedforward_dim=args.embed_dim, n_features=args.n_features)


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
    if new_model and finetune:
        print("Loading pretrained model")
        model_pretrained_dict = torch.load(
            "/s/chromatin/m/nobackup/ahmed/DeepPlant/results/results_DeepPlant_simple/255341/model_24_12_18:04:51.pt"
        )
        keys_pretrained = list(model_pretrained_dict.keys())[:-2]
        keys_net = list(network.state_dict())
        model_weights = network.state_dict()
        for i in range(len(keys_pretrained)):
            model_weights[keys_net[i]] = model_pretrained_dict[keys_pretrained[i]]
        network.load_state_dict(model_weights)
    return network
