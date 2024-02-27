"""Metrics for evaluating model performance."""

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def detach_and_pad(
    data: list[(Tensor, Tensor)], pad_token_id: int
) -> dict[str, Tensor]:
    """Detach tensors from accelerator and pad them to be the same length.

    Args:
        data (list[(Tensor, Tensor)]): A list of (prediction, target) tuples.
        pad_token_id (int): The ID of the padding token.
    """
    preds = [d[0].cpu().detach() for d in data]
    tgts = [d[1].cpu().detach() for d in data]

    max_pred_len = max([p.shape[1] for p in preds])
    min_pred_len = min([p.shape[1] for p in preds])
    max_tgt_len = max([t.shape[-1] for t in tgts])
    min_tgt_len = min([t.shape[-1] for t in tgts])

    if max_pred_len != min_pred_len:
        for idx, p in enumerate(preds):
            pred_pad_size = max_pred_len - p.shape[1]
            if pred_pad_size > 0:
                padding_logits = torch.ones_like(p[:, [0], :]) * float("-inf")
                padding_logits[:, :, pad_token_id] = 1.0
                padding_logits = torch.cat([padding_logits] * pred_pad_size, dim=1)
                preds[idx] = torch.cat((padding_logits, p), dim=1)

    if max_tgt_len != min_tgt_len:
        for idx, t in enumerate(tgts):
            tgt_pad_size = max_tgt_len - t.shape[-1]
            if tgt_pad_size > 0:
                t = F.pad(t, (tgt_pad_size, 0), mode="constant", value=pad_token_id)
                tgts[idx] = t

    try:
        preds = torch.cat(preds, dim=0)
        tgts = torch.cat(tgts, dim=0)
    except RuntimeError:
        print(preds.shape)
        print(tgts.shape)

    return {"predictions": preds, "targets": tgts}


def reduce_metrics(values: list[dict[str, any]]) -> dict[str, any]:
    """Returns a average of metrics weighted by sample count across batches.

    For each metric, computes the average across all batches weighted by the number
    of samples in each batch. The number of samples can vary depending on the kind of
    metric. For example, token-level metrics will have `batch_size * sequence_length`
    samples, while sequence-level metrics will have `batch_size` samples.

    Args:
        values (list[dict[str, any]]): A list of metrics for each batch. The list should
            be `num_batches` long, and each element is a dictionary of metrics. Each
            metric is itself a dictionary with keys `value` and `n_samples`.
    """
    metric_names = values[0].keys()
    weighted_metrics = {k: {"value": 0, "n_samples": 0} for k in metric_names}

    for v in values:
        for k in metric_names:
            weighted_metrics[k]["value"] += v[k]["value"] * v[k]["n_samples"]
            weighted_metrics[k]["n_samples"] += v[k]["n_samples"]

    for k in metric_names:
        weighted_metrics[k] = (
            weighted_metrics[k]["value"] / weighted_metrics[k]["n_samples"]
        )

    return weighted_metrics


def ce_loss(
    predictions: Tensor, targets: Tensor, ignore_index: int | None = None
) -> dict[str, float]:
    """Compute the cross-entropy loss.

    Args:
        predictions (Tensor): The predictions.
        targets (Tensor): The targets.
        ignore_index (int, optional): An index to ignore in the loss. Defaults to None.
    """
    targets = targets.flatten()
    predictions = predictions.flatten(end_dim=-2)
    value = (
        F.cross_entropy(
            input=predictions,
            target=targets,
            ignore_index=ignore_index,
        )
        .mean()
        .item()
    )

    return {
        "value": value,
        "n_samples": targets.size(0),
    }


def token_accuracy(
    predictions: Tensor, targets: Tensor, ignore_index: int | None = None
) -> dict[str, float]:
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
        predictions = predictions.argmax(dim=-1)

    assert predictions.size() == targets.size()

    value = (predictions == targets).float().mean().item()

    return {
        "value": value,
        "n_samples": targets.numel(),
    }


def sequence_accuracy(
    predictions: Tensor, targets: Tensor, ignore_index: int | None = None
) -> dict[str, float]:
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
        predictions = predictions.argmax(dim=-1)

    assert (
        predictions.size() == targets.size()
    ), f"{predictions.size()} != {targets.size()}"

    value = (
        (predictions == targets)
        .float()
        .sum(dim=1)
        .div(targets.size(1))
        .floor()
        .mean()
        .item()
    )

    return {
        "value": value,
        "n_samples": targets.size(0),
    }


# TODO: implement ignore_index
def mark_token_logprob(
    predictions: Tensor,
    targets: Tensor,
    ignore_index: int,
    mark_tok_id: int,
):
    """Return the logprob of the mark token in the prediction at the position preceding the mark token in the target."""  # noqa: E501
    mark_indices = (targets == mark_tok_id).nonzero(as_tuple=True)[1] - 1
    preds_before_mark = predictions[torch.arange(predictions.size(0)), mark_indices]

    # Calculate mean along batch dimension, i.e., the mean of what the model is
    # predicting right before the mark token, averaged over each sequence
    mean_preds_before_mark = preds_before_mark.mean(dim=0)

    # Logits to logprobs
    logprobs_before_mark = F.log_softmax(mean_preds_before_mark, dim=-1)

    # Find the logprob of the mark token
    mark_tok_logprob = logprobs_before_mark[mark_tok_id].item()

    return {
        "value": mark_tok_logprob,
        "n_samples": targets.size(0),
    }


def mark_token_prob(
    predictions: Tensor,
    targets: Tensor,
    ignore_index: int,
    mark_tok_id: int,
):
    """Return the probability of the mark token in the prediction at the position preceding the mark token in the target."""  # noqa: E501
    log_prob_dict = mark_token_logprob(predictions, targets, ignore_index, mark_tok_id)
    return {
        "value": np.exp(log_prob_dict["value"]),
        "n_samples": log_prob_dict["n_samples"],
    }


def compute_metrics(
    batch: list[(Tensor, Tensor)],
    pad_token_id: int,
    metric_fns: dict[str, Callable] = {
        "loss": ce_loss,
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
    },
    prefix: str | None = None,
) -> dict:
    """Compute metrics on a single batch of prediction-target pairs.

    Args:
        batch (list[(Tensor, Tensor)]): A list of (prediction, target) tuples.
        pad_token_id (int): The ID of the padding token.
        metric_fns (dict[str, Callable], optional): A dictionary of metric functions.
            Defaults to `ce_loss`, `token_accuracy`, and `sequence_accuracy`.
        prefix (str, optional): A prefix to add to the metric names. Defaults to None.
    """
    values_dict = {}

    data = detach_and_pad(batch, pad_token_id=pad_token_id)
    predicted_logits = data["predictions"]
    target_tokens = data["targets"]

    prefix_str = "" if prefix is None else f"{prefix}/"
    for metric_name, metric_fn in metric_fns.items():
        values_dict[prefix_str + metric_name] = metric_fn(
            predicted_logits, target_tokens, pad_token_id
        )

    return values_dict
