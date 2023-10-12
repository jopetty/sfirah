"""Metrics for evaluating model performance."""


from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def ce_loss(
    predictions: Tensor, targets: Tensor, ignore_index: int | None = None
) -> float:
    """Compute the cross-entropy loss.

    Args:
        predictions (Tensor): The predictions.
        targets (Tensor): The targets.
        ignore_index (int, optional): An index to ignore in the loss. Defaults to None.
    """
    targets = targets.flatten()
    predictions = predictions.flatten(end_dim=-2)
    return (
        F.cross_entropy(
            input=predictions,
            target=targets,
            ignore_index=ignore_index,
        )
        .mean()
        .item()
    )


def token_accuracy(
    predictions: Tensor, targets: Tensor, ignore_index: int | None = None
) -> float:
    """Compute the per-token accuracy of a batch of predictions.

    Computes the number of correctly-predicted tokens divided by the
    total number of tokens. The `predictions` can be passed in as logits
    or as tokens; if passed as logits, `argmax(dim=1)` will be applied.

    TODO: Implement `ignore_index`.

    Args:
        predictions (Tensor): The predictions.
        targets (Tensor): The targets.
        ignore_index (int, optional): An index to ignore in the loss. Defaults to None.
    """
    if predictions.dim() != targets.dim():
        predictions = predictions.argmax(dim=1)

    assert predictions.size() == targets.size()

    return (predictions == targets).float().mean().item()


def sequence_accuracy(
    predictions: Tensor, targets: Tensor, ignore_index: int | None = None
) -> float:
    """Compute the per-sequence accuracy of a batch of predictions.

    Computes the number of correctly-predicted sequences divided by the
    total number of sequences. The `predictions` can be passed in as logits
    or as tokens; if passed as logits, `argmax(dim=1)` will be applied.

    TODO: Implement `ignore_index`.

    Args:
        predictions (Tensor): The predictions.
        targets (Tensor): The targets.
        ignore_index (int, optional): An index to ignore in the loss. Defaults to None.
    """
    if predictions.dim() != targets.dim():
        predictions = predictions.argmax(dim=1)

    assert predictions.size() == targets.size()

    return (
        (predictions == targets)
        .float()
        .sum(dim=1)
        .div(targets.size(1))
        .floor()
        .mean()
        .item()
    )
