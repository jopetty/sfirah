"""Tests for the IDS4 model."""

import unittest

import torch

from sfirah import ids4


class TestIDS4TokenClassifier(unittest.TestCase):  # noqa: D101
    def test_init(self):  # noqa: D102
        _ = ids4.IDS4TokenClassifier(
            d_model=10, d_state=20, n_vocab=30, dropout=0.1, n_layers=1
        )

    def test_forward(self):  # noqa: D102
        model = ids4.IDS4TokenClassifier(
            d_model=10, d_state=20, n_vocab=30, dropout=0.1, n_layers=1
        )

        x = torch.ones([2, 4], dtype=torch.int64)
        _ = model(x)
