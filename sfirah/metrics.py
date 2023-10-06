"""Metrics for evaluating model performance."""


from torch import Tensor


def token_accuracy(predictions: Tensor, targets: Tensor) -> float:
    """Compute the per-token accuracy of a batch of predictions.

    Computes the number of correctly-predicted tokens divided by the
    total number of tokens. Note that `predictions` and `targets` must
    have the same shape---i.e., you should `argmax()` the predictions before
    passing them to this function.

    Args:
        predictions (Tensor): The predictions.
        targets (Tensor): The targets.
    """
    assert predictions.size() == targets.size()

    return (predictions == targets).float().mean().item()


def sequence_accuracy(predictions: Tensor, targets: Tensor) -> float:
    """Compute the per-sequence accuracy of a batch of predictions.

    Computes the number of correctly-predicted sequences divided by the
    total number of sequences. Note that `predictions` and `targets` must
    have the same shape---i.e., you should `argmax()` the predictions before
    passing them to this function.

    Args:
        predictions (Tensor): The predictions.
        targets (Tensor): The targets.
    """
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
