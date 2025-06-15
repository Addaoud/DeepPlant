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
        self.augment_n = 8

    def forward(
        self,
        y_pred,
        y_true,
    ):
        FeatDistLoss = 0
        if self.alpha != 0.0:
            FeatDistLoss = batched_cosine_distance(y_pred[1], group_size=self.augment_n)
        return (
            self.loss(
                y_pred[0],
                y_true.reshape(-1, y_true.shape[2]),
            )
            + self.alpha * FeatDistLoss
        )


def get_loss_fn(config):
    if config.loss.upper() == "MSE":
        return torch.nn.MSELoss()
    elif config.loss.upper() == "HUBER":
        return torch.nn.HuberLoss()
    else:
        return torch.nn.PoissonNLLLoss(log_input=False)
