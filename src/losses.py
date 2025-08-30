import torch.nn as nn
import torch
from typing import Optional, List, Tuple
import torch.nn.functional as F


def get_batched_triu_indices_cached(size: Tuple[int], device: str):
    """Return cached upper-triangular indices for size b x n x n (offset=1)."""
    if not hasattr(get_batched_triu_indices_cached, "_cache"):
        get_batched_triu_indices_cached._cache = {}
    key = (size, device)
    if key not in get_batched_triu_indices_cached._cache:
        get_batched_triu_indices_cached._cache[key] = torch.triu(
            torch.ones(size, device=device), diagonal=1
        )
    return get_batched_triu_indices_cached._cache[key]


def batched_cosine_distance(matrix_of_vectors, group_size):
    # matrix_of_vectors: [B * group_size, D]
    B = matrix_of_vectors.shape[0] // group_size
    matrix_of_vectors = matrix_of_vectors.view(B, group_size, -1)  # [B, augment_n, D]

    # Normalize
    x_norm = F.normalize(matrix_of_vectors, dim=-1)  # [B, augment_n, D]

    # Compute cosine similarity matrix: [B, augment_n, augment_n]
    sim = torch.bmm(x_norm, x_norm.transpose(1, 2))

    # Convert similarity to distance
    dist = 1 - sim

    # Optionally mask diagonal (distance with self = 0)
    # Sum upper triangle only (no duplicates or self-similarity)
    triu_mask = get_batched_triu_indices_cached(dist.size(), device=dist.device)
    loss = (dist * triu_mask).sum()

    return loss


class CustomCosineEmbeddingLoss(nn.Module):
    """
    custom semi-supervised loss for consistency regularization
    """

    def __init__(self, config):
        super(CustomCosineEmbeddingLoss, self).__init__()
        if config.loss.upper() == "MSE":
            self.loss = torch.nn.MSELoss(reduction="mean")
        elif config.loss.upper() == "HUBER":
            self.loss = torch.nn.HuberLoss()
        else:
            self.loss = torch.nn.PoissonNLLLoss(log_input=False)
        self.alpha = config.alpha
        try:
            self.consistency_regularization = config.consistency_regularization
        except:
            self.consistency_regularization = False
        self.augment_n = 6

    def forward(
        self,
        y_pred,
        y_true,
    ):
        FeatDistLoss = 0
        if (self.alpha != 0.0) and (self.consistency_regularization):
            FeatDistLoss = batched_cosine_distance(y_pred[1], group_size=self.augment_n)
            return (
                self.loss(
                    y_pred[0],
                    y_true.reshape(-1, y_true.shape[2]),
                )
                + self.alpha * FeatDistLoss
            )
        return self.loss(
            y_pred,
            y_true,
        )


class relative_mse_loss(nn.Module):
    def __init__(self, config, eps=1e-6):
        super(relative_mse_loss, self).__init__()
        self.eps = eps

    def forward(
        self,
        y_pred,
        y_true,
    ):
        return torch.mean(((y_pred - y_true) / (y_true + self.eps)) ** 2)


class MSLE(nn.Module):
    def __init__(self, config, eps=1e-6):
        super(relative_mse_loss, self).__init__()
        self.eps = eps

    def forward(
        self,
        y_pred,
        y_true,
    ):
        return (torch.log1p(y_pred + self.eps), torch.log1p(y_true + self.eps))


class PoissonNLLLoss(nn.Module):
    """
    custom PNLLloss that can be used with max of 2 genomes for now
    """

    def __init__(self, config):
        super(PoissonNLLLoss, self).__init__()
        self.PNLLloss = torch.nn.PoissonNLLLoss(log_input=False)
        self.alpha = config.alpha
        self.gamma = config.gamma

    def forward(
        self,
        y_pred,
        y_true,
    ):

        return self.PNLLloss(y_pred, y_true)


class MSE(nn.Module):
    def __init__(self, config):
        super(MSE, self).__init__()
        self.mse = torch.nn.MSELoss(reduction="mean")
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.nShared_filters = int(config.fraction * config.n_filters)

    def forward(
        self,
        y_pred,
        y_true,
        convlayers: Optional[List[torch.Tensor]] = [],
    ):
        if len(convlayers) > 1:
            return (
                self.mse(y_pred, y_true)
                + self.alpha
                * torch.norm(
                    convlayers[1][: self.nShared_filters, :, :]
                    - convlayers[0][: self.nShared_filters, :, :]
                )
                ** 2
                + self.gamma
                * torch.norm(
                    [convlayer for convlayer in convlayers if convlayer.requires_grad][
                        0
                    ]
                )
                ** 2
            )

        else:
            return self.mse(y_pred, y_true)


def get_loss_fn(config):
    if config.loss.upper() == "MSE":
        return MSE(config)
    elif config.loss.upper() == "HUBER":
        return torch.nn.HuberLoss()
    else:
        return PoissonNLLLoss(config)


class PoissonLoss(nn.Module):
    def __init__(self):
        super(PoissonLoss, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean(y_pred - y_true * torch.log(y_pred + 1e-8))


class BasenjiPearsonR(nn.Module):
    def __init__(self):
        super(BasenjiPearsonR, self).__init__()

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()

        product = torch.sum(y_true * y_pred, dim=[0, 1])
        true_sum = torch.sum(y_true, dim=[0, 1])
        true_sumsq = torch.sum(y_true**2, dim=[0, 1])
        pred_sum = torch.sum(y_pred, dim=[0, 1])
        pred_sumsq = torch.sum(y_pred**2, dim=[0, 1])
        count = torch.ones_like(y_true).sum(dim=[0, 1])
        true_mean = true_sum / count
        true_mean2 = true_mean**2
        pred_mean = pred_sum / count
        pred_mean2 = pred_mean**2

        term1 = product
        term2 = -true_mean * pred_sum
        term3 = -pred_mean * true_sum
        term4 = count * true_mean * pred_mean
        covariance = term1 + term2 + term3 + term4

        true_var = true_sumsq - count * true_mean2
        pred_var = pred_sumsq - count * pred_mean2
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var

        return -torch.mean(correlation)


class pearsonr_poisson(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        if not self.alpha:
            print("ALPHA SET TO DEFAULT VALUE!")
            self.alpha = 0.1  ### TODO: SET TO 0.001

    def forward(self, y_true, y_pred):
        # multinomial part of loss function
        pr_loss = BasenjiPearsonR()
        pr = pr_loss(y_true, y_pred)
        # poisson part
        poiss_loss = PoissonLoss()
        poiss = poiss_loss(y_true, y_pred)
        # sum with weight
        total_loss = (2 * pr * poiss) / (pr + poiss)
        return total_loss
