"""Loss functions used by the risk classifier."""

import torch
from torch import Tensor, nn


class WeightedFocalLoss(nn.Module):
    """Binary focal loss with a separate weight for positive examples."""

    def __init__(self, positive_weight: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        if positive_weight <= 0:
            raise ValueError("positive_weight must be greater than zero")
        if gamma < 0:
            raise ValueError("gamma cannot be negative")
        self.positive_weight = positive_weight
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.float().reshape_as(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = targets * self.positive_weight + (1 - targets)
        focal_factor = (1 - p_t).clamp_min(1e-8).pow(self.gamma)
        return (alpha_t * focal_factor * bce).mean()
