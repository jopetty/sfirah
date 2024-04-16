"""Tests for the IDS4 model."""

import unittest
from itertools import accumulate

import torch

from sfirah import ids4


class TestIDS4TokenClassifier(unittest.TestCase):  # noqa: D101
    def test_init(self):  # noqa: D102
        _ = ids4.IDS4TokenClassifier(
            d_model=10,
            d_state=20,
            n_vocab=30,
            dropout=0.1,
            n_layers=1,
            activation="gelu",
        )

    def test_forward(self):  # noqa: D102
        model = ids4.IDS4TokenClassifier(
            d_model=10,
            d_state=20,
            n_vocab=30,
            dropout=0.1,
            n_layers=1,
            activation="gelu",
        )

        x = torch.ones([2, 4], dtype=torch.int64)
        _ = model(x)

    def test_fast_forward(self):  # noqa: D102
        model = ids4.IDS4TokenClassifier(
            d_model=5, d_state=6, n_vocab=30, dropout=0.0, n_layers=1, activation="gelu"
        )

        x = torch.ones([2, 4], dtype=torch.int64)
        _ = model(x)

    def check_fast_slow_equivalent(self):
        """Check if the outputs of a multi-layer IDS4 model are equivalent.

        Uses the actualy implementation of the IDS4 fast and slow forward passes.
        """
        model = ids4.IDS4TokenClassifier(
            d_model=5, d_state=6, n_vocab=30, dropout=0.0, n_layers=1, activation="gelu"
        )

        x = torch.ones([2, 4], dtype=torch.int64)

        with torch.no_grad():
            x_slow = model.forward_slow(x)
            x_fast = model(x)

        self.assertTrue(torch.allclose(x_fast, x_slow))

    def test_cum_bmm(self):
        """Manual test for cummulative bmm operation.

        This exists as a sanity check to ensure that the fast and slow versions
        of the IDS4 matrix multiplication are equivalent.
        """
        ax = torch.randn(2, 4, 6, 6)
        proj = torch.nn.Linear(6, 1)

        cum_prod_slow = []
        for i in range(ax.shape[0]):
            ax_i = torch.unbind(ax[i], dim=0)
            prod_list = list(accumulate(ax_i, lambda x, y: x @ y))
            prod_list = [proj(p) for p in prod_list]
            ax_i = torch.stack(prod_list, dim=0).squeeze()
            cum_prod_slow.append(ax_i)
        cum_prod_slow = torch.stack(cum_prod_slow, dim=0).squeeze()

        ax_i = torch.unbind(ax, dim=1)
        cum_prod_fast = list(accumulate(ax_i, lambda x, y: torch.bmm(x, y)))
        cum_prod_fast = [proj(p) for p in cum_prod_fast]
        cum_prod_fast = torch.stack(cum_prod_fast, dim=1).squeeze()

        self.assertTrue(torch.allclose(cum_prod_fast, cum_prod_slow))
