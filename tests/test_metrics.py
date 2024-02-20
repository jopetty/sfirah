"""Tests for the metrics module."""


import unittest

from torch import Tensor

from sfirah import metrics


class TestMetrics(unittest.TestCase):  # noqa: D101
    def test_token_accuracy(self):  # noqa: D102
        # Single-sequence batch w/ perfect accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 2, 3, 4]])
        tok_acc = {"value": 1.0, "n_samples": 4}
        self.assertEqual(metrics.token_accuracy(predictions, targets), tok_acc)

        # Single-sequence batch w/ half accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 2, 2, 1]])
        tok_acc = {"value": 0.5, "n_samples": 4}
        self.assertEqual(metrics.token_accuracy(predictions, targets), tok_acc)

        # Single-sequence batch w/ only one correct token
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 1, 1, 1]])
        tok_acc = {"value": 0.25, "n_samples": 4}
        self.assertEqual(metrics.token_accuracy(predictions, targets), tok_acc)

        # Single-sequence batch w/ zero accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[4, 3, 2, 1]])
        tok_acc = {"value": 0.0, "n_samples": 4}
        self.assertEqual(metrics.token_accuracy(predictions, targets), tok_acc)

        # Multi-sequence batch w/ perfect accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        tok_acc = {"value": 1.0, "n_samples": 8}
        self.assertEqual(metrics.token_accuracy(predictions, targets), tok_acc)

        # Multi-sequence batch w/ half accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 2, 3, 4], [2, 2, 2, 2]])
        tok_acc = {"value": 0.5, "n_samples": 8}
        self.assertEqual(metrics.token_accuracy(predictions, targets), tok_acc)

        # Multi-sequence batch w/ zero accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[4, 3, 2, 1], [2, 2, 2, 2]])
        tok_acc = {"value": 0.0, "n_samples": 8}
        self.assertEqual(metrics.token_accuracy(predictions, targets), tok_acc)

        # Multi-sequence batch w/ only one correct token
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
        tok_acc = {"value": 0.125, "n_samples": 8}
        self.assertEqual(metrics.token_accuracy(predictions, targets), tok_acc)

        # Single-sequence batch w/ single-token sequence
        predictions = Tensor([[1]])
        targets = Tensor([[1]])
        tok_acc = {"value": 1.0, "n_samples": 1}
        self.assertEqual(metrics.token_accuracy(predictions, targets), tok_acc)

    def test_sequence_accuracy(self):  # noqa: D102
        # Single-sequence batch w/ perfect accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 2, 3, 4]])
        seq_acc = {"value": 1.0, "n_samples": 1}
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), seq_acc)

        # Single-sequence batch w/ half accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 2, 2, 1]])
        seq_acc = {"value": 0.0, "n_samples": 1}
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), seq_acc)

        # Single-sequence batch w/ only one correct token
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 1, 1, 1]])
        seq_acc = {"value": 0.0, "n_samples": 1}
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), seq_acc)

        # Single-sequence batch w/ zero accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[4, 3, 2, 1]])
        seq_acc = {"value": 0.0, "n_samples": 1}
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), seq_acc)

        # Multi-sequence batch w/ perfect accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        seq_acc = {"value": 1.0, "n_samples": 2}
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), seq_acc)

        # Multi-sequence batch w/ half accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 2, 3, 4], [2, 2, 2, 2]])
        seq_acc = {"value": 0.5, "n_samples": 2}
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), seq_acc)

        # Multi-sequence batch w/ zero accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[4, 3, 2, 1], [2, 2, 2, 2]])
        seq_acc = {"value": 0.0, "n_samples": 2}
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), seq_acc)

        # Multi-sequence batch w/ only one correct token
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
        seq_acc = {"value": 0.0, "n_samples": 2}
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), seq_acc)

        # Single-sequence batch w/ single-token sequence
        predictions = Tensor([[1]])
        targets = Tensor([[1]])
        seq_acc = {"value": 1.0, "n_samples": 1}
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), seq_acc)

    def test_mark_token_logprob(self):  # noqa: D102
        mark_tok_id = 0

        targets = Tensor([[1, 2, 3, 0], [1, 1, 1, 0], [3, 2, 0, 1]])

        predictions = Tensor(
            [
                [
                    [0.1, 0.8, 0.05, 0.05],
                    [0.1, 0.3, 0.4, 0.6],
                    [0.3, 0.5, 0.4, 0.3],
                    [0.1, 0.3, 0.4, 0.1],
                ],
                [
                    [0.7, 0.5, 0.15, 0.45],
                    [0.6, 0.3, 0.5, 0.1],
                    [0.0, 0.1, 0.3, 0.4],
                    [0.4, 0.5, 0.3, 0.2],
                ],
                [
                    [0.1, 0.7, 0.6, 0.02],
                    [0.1, 0.7, 0.44, 0.7],
                    [0.3, 0.4, 0.6, 0.2],
                    [0.11, 0.2, 0.1, 0.12],
                ],
            ]
        )

        ground_truth = {"value": 0.1333, "n_samples": 3}
        calculated = metrics.mark_token_logprob(predictions, targets, mark_tok_id)

        self.assertEqual(ground_truth["n_samples"], calculated["n_samples"])
        self.assertAlmostEqual(ground_truth["value"], calculated["value"], places=4)


if __name__ == "__main__":
    unittest.main()
