import torch

from src.losses import WeightedFocalLoss


def test_focal_loss_returns_scalar_and_gradient() -> None:
    logits = torch.tensor([[0.0], [1.0]], requires_grad=True)
    targets = torch.tensor([[0.0], [1.0]])

    loss = WeightedFocalLoss(positive_weight=2.0)(logits, targets)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None