"""Tests for Flax models."""


import unittest

from sfirah import flax_transformer


class TestFlaxCausalLM(unittest.TestCase):  # noqa: D101
    def test_init(self):  # noqa: D102
        config = flax_transformer.TransformerConfig(
            vocab_size=10,
            output_vocab_size=10,
        )

        print(config)

        model = flax_transformer.CausalLM(config=config)
        print(model)
