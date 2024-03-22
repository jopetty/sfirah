"""Tests for models."""


import unittest

import torch
from torch.nn import functional as F  # noqa: N812

from sfirah import s4, transformers


class TestS4TokenClassifier(unittest.TestCase):  # noqa: D101
    def test_init(self):  # noqa: D102
        _ = s4.S4TokenClassifier(
            d_model=10,
            d_state=20,
            n_vocab=30,
            dropout=0.1,
            transposed=True,
            n_layers=2,
            prenorm=False,
            lr=0.1,
        )


class TestEncoderTokenClassifer(unittest.TestCase):  # noqa: D101
    def test_init(self):  # noqa: D102
        _ = transformers.EncoderTokenClassifier(
            d_model=10,
            d_ff=20,
            dropout=0.1,
            activation="relu",
            n_layers=2,
            n_heads=2,
            norm_first=False,
            batch_first=True,
            n_vocab=30,
            weight_sharing=True,
            bias=True,
            layer_norm_eps=1e-5,
            weight_scale=1.0,
        )

    def test_forward(self):  # noqa: D102
        batch_size = 2
        seq_len = 4
        n_vocab = 30

        model = transformers.EncoderTokenClassifier(
            d_model=10,
            d_ff=20,
            dropout=0.1,
            activation="relu",
            n_layers=2,
            n_heads=2,
            norm_first=False,
            batch_first=True,
            n_vocab=n_vocab,
            weight_sharing=True,
            bias=True,
            layer_norm_eps=1e-5,
            weight_scale=1.0,
        )

        x = torch.ones([batch_size, seq_len], dtype=torch.int64)
        y = model(x)

        self.assertEqual(y.shape, torch.Size([batch_size, seq_len, n_vocab]))
        self.assertFalse(torch.isnan(y).any())

    def test_ce(self):  # noqa: D102
        batch_size = 2
        seq_len = 4
        n_vocab = 30

        model = transformers.EncoderTokenClassifier(
            d_model=10,
            d_ff=20,
            dropout=0.1,
            activation="relu",
            n_layers=2,
            n_heads=2,
            norm_first=False,
            batch_first=True,
            n_vocab=n_vocab,
            weight_sharing=True,
            bias=True,
            layer_norm_eps=1e-5,
            weight_scale=1.0,
        )

        x = torch.ones([batch_size, seq_len], dtype=torch.int64)
        y = x
        y_hat = model(x)

        y_hat = torch.flatten(y_hat, start_dim=0, end_dim=-2)
        y = torch.flatten(y)

        F.cross_entropy(y_hat, y)


class TestEncoderSequenceClassifier(unittest.TestCase):  # noqa: D101
    def test_init(self):  # noqa: D102
        _ = transformers.EncoderSequenceClassifier(
            cl_dim=1,
            cl_index=0,
            d_model=10,
            d_ff=20,
            dropout=0.1,
            activation="relu",
            n_layers=2,
            n_heads=2,
            norm_first=False,
            batch_first=True,
            n_vocab=30,
            weight_sharing=True,
            bias=True,
            layer_norm_eps=1e-5,
            weight_scale=1.0,
        )

    def test_forward(self):  # noqa: D102
        batch_size = 2
        seq_len = 4
        n_vocab = 30

        model = transformers.EncoderSequenceClassifier(
            cl_dim=1,
            cl_index=0,
            d_model=10,
            d_ff=20,
            dropout=0.1,
            activation="relu",
            n_layers=2,
            n_heads=2,
            norm_first=False,
            batch_first=True,
            n_vocab=n_vocab,
            weight_sharing=True,
            bias=True,
            layer_norm_eps=1e-5,
            weight_scale=1.0,
        )

        x = torch.ones([batch_size, seq_len], dtype=torch.int64)
        y = model(x)

        print(x.shape)
        print(y.shape)
        print(model)

        self.assertEqual(y.shape, torch.Size([batch_size, n_vocab]))
        self.assertFalse(torch.isnan(y).any())


class TestCausalDecoder(unittest.TestCase):  # noqa: D101
    def test_init(self):  # noqa: D102
        _ = transformers.CausalDecoder(
            block_size=128,
            d_model=10,
            d_ff=20,
            dropout=0.1,
            activation="relu",
            n_layers=2,
            n_heads=2,
            norm_first=False,
            batch_first=True,
            n_vocab=30,
            weight_sharing=True,
            bias=True,
            layer_norm_eps=1e-5,
            weight_scale=1.0,
        )

    def test_generate(self):  # noqa: D102
        model = transformers.CausalDecoder(
            block_size=128,
            d_model=10,
            d_ff=20,
            dropout=0.1,
            activation="relu",
            n_layers=2,
            n_heads=2,
            norm_first=False,
            batch_first=True,
            n_vocab=10,
            weight_sharing=True,
            bias=True,
            layer_norm_eps=1e-5,
            weight_scale=1.0,
        )

        x = torch.ones([1, 11], dtype=torch.int64)

        y = model.generate(x, max_length=21, temperature=0.0, eos_token_id=5)

        print(x)
        print(y)


class TestGPT(unittest.TestCase):  # noqa: D101
    def test_init(self):  # noqa: D102
        _ = transformers.GPT(
            bias=False,
            block_size=128,
            d_model=10,
            d_ff=20,
            dropout=0.1,
            n_layers=2,
            n_heads=2,
            n_vocab=30,
        )

    def test_generate(self):  # noqa: D102
        model = transformers.GPT(
            bias=False,
            block_size=128,
            d_model=10,
            d_ff=20,
            dropout=0.1,
            n_layers=2,
            n_heads=2,
            n_vocab=30,
        )

        x = torch.ones([1, 11], dtype=torch.int64)

        y = model.generate(x, max_length=21, temperature=0.0, eos_token_id=5)

        print(x)
        print(y)


if __name__ == "__main__":
    unittest.main()
