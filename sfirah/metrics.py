from torch import Tensor


def token_accuracy(predictions: Tensor, targets: Tensor) -> float:
    assert predictions.size() == targets.size()

    return (predictions == targets).float().mean().item()


def sequence_accuracy(predictions: Tensor, targets: Tensor) -> float:
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
