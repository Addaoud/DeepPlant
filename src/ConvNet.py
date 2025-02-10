import torch.nn as nn
import torch


class Residual(nn.Module):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)


class ConvNet(nn.Module):
    def __init__(self, n_filters: int, n_genomes: int):
        super(ConvNet, self).__init__()
        self.n_filters = n_filters
        self.n_genomes = n_genomes
        self.conv_features = list()
        for _ in range(self.n_genomes):
            self.conv_features.append(
                nn.Sequential(
                    nn.Conv1d(4, self.n_filters, kernel_size=13, padding=6),
                    nn.ReLU(inplace=True),
                )
            )
        self.conv_features = nn.ModuleList(self.conv_features)

        self.conv_network = nn.Sequential(
            nn.Conv1d(self.n_filters, 2 * self.n_filters, kernel_size=11, padding=5),
            nn.BatchNorm1d(2 * self.n_filters),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(2 * self.n_filters, 3 * self.n_filters, kernel_size=9, padding=4),
            nn.BatchNorm1d(3 * self.n_filters),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.num_channels = 3 * self.n_filters

    def forward(self, input: torch.Tensor, bit: int = 0):
        features = self.conv_features[bit](input)
        return self.conv_network(features)


def build_ConvNet(args):
    if len(args.n_features) > 1:
        n_filters = len(args.n_features) * args.n_filters
    else:
        n_filters = args.n_filters
    convnet = ConvNet(n_filters=n_filters, n_genomes=len(args.n_features))
    if args.use_pretrained_filter:
        filters_list = [
            torch.load(model_path, map_location=torch.device("cpu"))[
                "module.backbone.conv_network.0.weight"
            ]
            for model_path in args.models_list
        ]
        with torch.no_grad():
            for i in range(2):
                for j, conv_layer in zip(range(2), filters_list):
                    convnet.conv_features[i][0].weight[
                        j
                        * (n_filters // len(args.n_features)) : (j + 1)
                        * (n_filters // len(args.n_features)),
                        :,
                        :,
                    ] = conv_layer
    return convnet


# def build_ConvNet(args):
#    return ConvNet(n_filters=args.n_filters, n_genomes=len(args.n_features))
