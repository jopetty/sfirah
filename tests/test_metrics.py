"""Tests for the metrics module."""


import unittest

from torch import Tensor

from sfirah import metrics


class TestMetrics(unittest.TestCase):  # noqa: D101
    def test_token_accuracy(self):  # noqa: D102
        # Single-sequence batch w/ perfect accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 2, 3, 4]])
        self.assertEqual(metrics.token_accuracy(predictions, targets), 1.0)

        # Single-sequence batch w/ half accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 2, 2, 1]])
        self.assertEqual(metrics.token_accuracy(predictions, targets), 0.5)

        # Single-sequence batch w/ only one correct token
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 1, 1, 1]])
        self.assertEqual(metrics.token_accuracy(predictions, targets), 0.25)

        # Single-sequence batch w/ zero accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[4, 3, 2, 1]])
        self.assertEqual(metrics.token_accuracy(predictions, targets), 0.0)

        # Multi-sequence batch w/ perfect accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        self.assertEqual(metrics.token_accuracy(predictions, targets), 1.0)

        # Multi-sequence batch w/ half accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 2, 3, 4], [2, 2, 2, 2]])
        self.assertEqual(metrics.token_accuracy(predictions, targets), 0.5)

        # Multi-sequence batch w/ zero accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[4, 3, 2, 1], [2, 2, 2, 2]])
        self.assertEqual(metrics.token_accuracy(predictions, targets), 0.0)

        # Multi-sequence batch w/ only one correct token
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
        self.assertEqual(metrics.token_accuracy(predictions, targets), 0.125)

        # Single-sequence batch w/ single-token sequence
        predictions = Tensor([[1]])
        targets = Tensor([[1]])
        self.assertEqual(metrics.token_accuracy(predictions, targets), 1.0)

    def test_sequence_accuracy(self):  # noqa: D102
        # Single-sequence batch w/ perfect accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 2, 3, 4]])
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), 1.0)

        # Single-sequence batch w/ half accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 2, 2, 1]])
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), 0.0)

        # Single-sequence batch w/ only one correct token
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[1, 1, 1, 1]])
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), 0.0)

        # Single-sequence batch w/ zero accuracy
        predictions = Tensor([[1, 2, 3, 4]])
        targets = Tensor([[4, 3, 2, 1]])
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), 0.0)

        # Multi-sequence batch w/ perfect accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), 1.0)

        # Multi-sequence batch w/ half accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 2, 3, 4], [2, 2, 2, 2]])
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), 0.5)

        # Multi-sequence batch w/ zero accuracy
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[4, 3, 2, 1], [2, 2, 2, 2]])
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), 0.0)

        # Multi-sequence batch w/ only one correct token
        predictions = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        targets = Tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), 0.0)

        # Single-sequence batch w/ single-token sequence
        predictions = Tensor([[1]])
        targets = Tensor([[1]])
        self.assertEqual(metrics.sequence_accuracy(predictions, targets), 1.0)


if __name__ == "__main__":
    unittest.main()
