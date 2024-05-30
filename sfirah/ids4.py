"""Input-dependent S4 (IDS4)."""

from itertools import accumulate

import numpy as np
import torch
from einops import einsum
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class IDS4Block(nn.Module):
    """Input-dependent S4 Block."""

    @property
    def d_state(self) -> int:  # noqa: D102
        return self._d_state

    @property
    def d_model(self) -> int:  # noqa: D102
        return self._d_model

    def make_HiPPO(self, N):  # noqa: N802, N803
        """Construct HiPPO matrix for initializing self.A."""
        P = np.sqrt(1 + 2 * np.arange(N))  # noqa: N806
        A = P[:, np.newaxis] * P[np.newaxis, :]  # noqa: N806
        A = np.tril(A) - np.diag(np.arange(N))  # noqa: N806
        return -A

    def __init__(  # noqa: D107
        self, d_state: int, d_model: int
    ):
        super().__init__()

        self._d_state = d_state
        self._d_model = d_model

        # hippo = torch.tensor(self.make_HiPPO(d_state), dtype=torch.float32)
        # self.A = nn.Parameter(repeat(hippo, "i j -> b i j", b=d_model))

        rand_idm = torch.eye(self.d_state).unsqueeze(0) + torch.randn(
            self.d_model, self.d_state, self.d_state
        ) / torch.sqrt(torch.tensor(self.d_state))
        self.A = nn.Parameter(rand_idm)
        self.proj = nn.Linear(self.d_state, 1)
        self.C = nn.Linear(self.d_state, self.d_model)
        self.D = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, L, d_model)
        """
        Ax = einsum(x, self.A, "b l dm, dm ds dst -> b l ds dst")  # noqa: N806
        Ax = F.gelu(Ax)  # noqa: N806  (B, L, d_state, d_state)

        Ax_i = torch.unbind(Ax, dim=1)  # noqa: N806
        cum_prod = list(accumulate(Ax_i, lambda x, y: torch.bmm(x, y)))
        cum_prod = [self.proj(p) for p in cum_prod]
        cum_prod = torch.stack(cum_prod, dim=1).squeeze()  # noqa: N806

        return self.C(cum_prod) + self.D(x)


class IDS4TokenClassifier(nn.Module):
    """IDS4 Token Classifier."""

    @property
    def d_model(self) -> int:  # noqa: D102
        return self._d_model

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self._n_vocab

    @property
    def d_state(self) -> int:  # noqa: D102
        return self._d_state

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def dropout(self) -> float:  # noqa: D102
        return self._dropout

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def prenorm(self) -> bool:  # noqa: D102
        return self._prenorm

    @property
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(  # noqa: D107
        self,
        d_model: int,
        d_state: int,
        n_layers: int,
        n_vocab: int,
        dropout: float,
        bias: bool = True,
        prenorm: bool = False,
    ):
        super().__init__()

        self._bias = bias
        self._d_model = d_model
        self._n_vocab = n_vocab
        self._d_state = d_state
        self._n_layers = n_layers
        self._dropout = dropout
        self._prenorm = prenorm

        self.embedding = nn.Embedding(self.n_vocab, self.d_model)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(IDS4Block(d_model=self.d_model, d_state=self.d_state))
            self.norms.append(nn.LayerNorm(self.d_model))
            self.dropout_layers.append(nn.Dropout(self.dropout))

        self.cl_head = nn.Linear(self.d_model, self.n_vocab, self.bias)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.embedding(x)  # (B, L, d_input) -> (B, L, d_model)
        # x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(
            self.s4_layers, self.norms, self.dropout_layers
        ):
            z = x
            if self.prenorm:
                z = norm(z)

            z = layer(z)
            z = dropout(z)
            x = x + z

            if not self.prenorm:
                x = norm(x)

        x = self.cl_head(x)

        return x
