import torch.nn as nn
import torch
from typing import Tuple
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
    matrix_of_vectors = matrix_of_vectors.reshape(
        B, group_size, -1
    )  # [B, augment_n, D]

    # Normalize
    x_norm = F.normalize(matrix_of_vectors, dim=-1, eps=1e-6)  # [B, augment_n, D]

    # Compute cosine similarity matrix: [B, augment_n, augment_n]
    sim = torch.bmm(x_norm, x_norm.transpose(1, 2))

    # Convert similarity to distance
    dist = 1 - sim

    # Optionally mask diagonal (distance with self = 0)
    # Sum upper triangle only (no duplicates or self-similarity)
    triu_mask = get_batched_triu_indices_cached(dist.size(), device=dist.device)
    loss = (dist * triu_mask).sum() / B
    return loss


def batched_L2_distance(matrix_of_vectors, group_size):
    B = matrix_of_vectors.shape[0] // group_size
    matrix_of_vectors = matrix_of_vectors.view(B, group_size, -1)  # [B, augment_n, D]
    dist = torch.cdist(matrix_of_vectors, matrix_of_vectors, p=2)
    triu_mask = get_batched_triu_indices_cached(dist.size(), device=dist.device)
    loss = (dist * triu_mask).sum() / B
    return loss


class relative_mse_loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(relative_mse_loss, self).__init__()
        self.eps = eps

    def forward(
        self,
        y_pred,
        y_true,
    ):
        return torch.mean(((y_pred - y_true) / (y_true + self.eps)) ** 2)


def per_gene_corr_loss(pred, target, eps=1e-8):
    # pred, target: (G, C) — G genes, C conditions each (same conditions or not)
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = pred.norm(dim=1) * target.norm(dim=1) + eps
    return (1 - num / den).mean()  # 1 - r, averaged over genes


class expression_loss(nn.Module):
    def __init__(self, eps=1e-6, lam=1e-1):
        super().__init__()
        self.eps = eps
        self.lam = lam
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, y_pred, y_true):
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        assert (
            y_pred.shape == y_true.shape and y_pred.dim() == 2
        ), "expected (genes, conditions)"
        return self.mse(y_pred, y_true)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha: per-class weight tensor (like CE's weight), or None
        self.register_buffer(
            "alpha",
            torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None,
        )

    def forward(self, logits, target):
        # logits: (N, C), target: (N,) class indices
        ce = torch.nn.functional.cross_entropy(
            logits, target, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce)  # prob of the true class
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_fn(config):
    if config.loss.upper() == "MSE":
        return torch.nn.MSELoss(reduction="mean")
    elif config.loss.upper() == "HUBER":
        return torch.nn.SmoothL1Loss(
            reduction="mean"
        )  # torch.nn.HuberLoss(reduction="mean")
    elif config.loss.upper() == "RMSE":
        return relative_mse_loss()
    elif config.loss.upper() == "BCE":
        return torch.nn.BCEWithLogitsLoss()
    elif config.loss.upper() == "CE":
        return torch.nn.CrossEntropyLoss(reduction="mean")
    elif config.loss.upper() == "WCE":
        weight = torch.tensor(config.class_weights, dtype=torch.float32)
        return torch.nn.CrossEntropyLoss(weight=weight, reduction="mean")
    elif config.loss.upper() == "FOCAL":
        alpha = getattr(config, "focal_alpha", None)  # per-class weights, optional
        gamma = getattr(config, "focal_gamma", 2.0)
        return FocalLoss(gamma=gamma, alpha=alpha, reduction="mean")
    elif config.loss.upper() == "POISSON":
        return torch.nn.PoissonNLLLoss(log_input=False)
    else:
        raise Exception("Loss function not found")


class CustomEmbeddingLoss(nn.Module):
    """
    custom semi-supervised loss for consistency regularization
    """

    def __init__(self, config):
        super(CustomEmbeddingLoss, self).__init__()
        self.loss = get_loss_fn(config)
        self.alpha = config.alpha
        self.dist = config.dist
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
        if self.consistency_regularization:
            if self.alpha != 0.0:
                if self.dist == "cosine":
                    FeatDistLoss = batched_cosine_distance(
                        y_pred[1], group_size=self.augment_n
                    )
                else:
                    FeatDistLoss = batched_L2_distance(
                        y_pred[1], group_size=self.augment_n
                    )
                return (
                    self.loss(
                        y_pred[0],
                        y_true.reshape(-1, y_true.shape[2]),
                    )
                    + self.alpha * FeatDistLoss
                )
            else:
                return self.loss(
                    y_pred[0],
                    y_true.reshape(-1, y_true.shape[2]),
                )
        return self.loss(
            y_pred,
            y_true,
        )


class CrossEntropyWithL1(nn.Module):
    def __init__(self, model, l1_lambda=1e-5):
        """
        model: the neural network (nn.Module)
        l1_lambda: regularization strength
        """
        super().__init__()
        self.model = model
        self.l1_lambda = l1_lambda
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        # Standard cross-entropy loss
        ce = self.ce_loss(outputs, targets)

        # L1 regularization term
        l1_norm = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                l1_norm += torch.sum(torch.abs(param))

        # Total loss
        loss = ce + self.l1_lambda * l1_norm
        return loss


class BCEWithL1(nn.Module):
    def __init__(self, model, l1_lambda=1e-5):
        """
        model: the neural network (nn.Module)
        l1_lambda: regularization strength
        """
        super().__init__()
        self.model = model
        self.l1_lambda = l1_lambda
        self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # Standard cross-entropy loss
        ce = self.ce_loss(outputs, targets.float())

        # L1 regularization term
        l1_norm = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                l1_norm += torch.sum(torch.abs(param))

        # Total loss
        loss = ce + self.l1_lambda * l1_norm
        return loss


class LLMRegressionLoss(nn.Module):
    """
    Supervised regression loss for the LLM benchmark. Drop-in replacement for
    CustomEmbeddingLoss, but with the consistency-regularization path removed:
    the LLM regressor returns a single (B, n_tasks) tensor against (B, n_tasks)
    targets, so no augmentation grouping or reshaping is needed.
    """

    def __init__(self, config):
        super().__init__()
        self.loss = get_loss_fn(config)

    def forward(self, y_pred, y_true):
        # LLM regressor never returns a tuple; guard anyway in case a backbone
        # hands back (logits, hidden_states).
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        return self.loss(y_pred, y_true)
