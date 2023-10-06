"""Metrics for evaluating model performance."""


from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def ce_loss(predictions: Tensor, targets: Tensor) -> float:
    """Compute the cross-entropy loss.

    Args:
        predictions (Tensor): The predictions.
        targets (Tensor): The targets.
    """
    return F.cross_entropy(predictions, targets).mean().item()


def token_accuracy(predictions: Tensor, targets: Tensor) -> float:
    """Compute the per-token accuracy of a batch of predictions.

    Computes the number of correctly-predicted tokens divided by the
    total number of tokens. The `predictions` can be passed in as logits
    or as tokens; if passed as logits, `argmax(dim=1)` will be applied.

    Args:
        predictions (Tensor): The predictions.
        targets (Tensor): The targets.
    """
    if predictions.dim() != targets.dim():
        predictions = predictions.argmax(dim=1)

    assert predictions.size() == targets.size()

    return (predictions == targets).float().mean().item()


def sequence_accuracy(predictions: Tensor, targets: Tensor) -> float:
    """Compute the per-sequence accuracy of a batch of predictions.

    Computes the number of correctly-predicted sequences divided by the
    total number of sequences. The `predictions` can be passed in as logits
    or as tokens; if passed as logits, `argmax(dim=1)` will be applied.

    Args:
        predictions (Tensor): The predictions.
        targets (Tensor): The targets.
    """
    if predictions.dim() != targets.dim():
        predictions = predictions.argmax(dim=1)

    assert predictions.size() == targets.size()
    assert predictions.dim() == 2

    return (
        (predictions == targets)
        .float()
        .sum(dim=1)
        .div(targets.size(1))
        .floor()
        .mean()
        .item()
    )
