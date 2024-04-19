"""Unit tests for NC1-attention transformers."""

import math
import unittest
from functools import reduce

import torch


class TestNC1Attention(unittest.TestCase):  # noqa: D101
    def test_simple_matmul(self):  # noqa: D102
        # Test case with B, H = 1 so we can ignore them
        T, D = 4, 5  # noqa: N806
        # make q,k random integer tensors of size TD with integer values
        q = torch.randint(0, 10, (T, D)).float()
        k = torch.randint(0, 10, (T, D)).float()
        v = torch.randint(0, 10, (T, D)).float()

        attn = torch.einsum("id,je -> ijde", q, k)
        attn = torch.unbind(attn, dim=1)
        attn = reduce(
            lambda x, y: torch.einsum("abc,acf -> abf", x, y),  # bmm w/ einsum
            attn,
        )
        attn *= 1.0 / math.sqrt(k.size(-1))
        print(attn.size())
        print(v.size())

        attn @ v
