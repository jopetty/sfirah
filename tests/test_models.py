"""Tests for models."""


import unittest

import torch

from sfirah import transformers


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

        self.assertEqual(y.shape, torch.Size([batch_size, n_vocab, seq_len]))
        self.assertFalse(torch.isnan(y).any())


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
            context_size=128,
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


if __name__ == "__main__":
    unittest.main()
