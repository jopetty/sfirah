"""Input-dependent S4 (IDS4)."""

from itertools import accumulate

import torch
from einops import einsum
from torch import Tensor, nn

from .utils import get_activation


class IDS4Block(nn.Module):
    """Input-dependent S4 Block."""

    @property
    def d_state(self) -> int:  # noqa: D102
        return self._d_state

    @property
    def d_model(self) -> int:  # noqa: D102
        return self._d_model

    @property
    def activation(self) -> str:  # noqa: D102
        return self._activation

    def __init__(  # noqa: D107
        self, activation: str, d_state: int, d_model: int
    ):
        super().__init__()

        self._activation = activation
        self._d_state = d_state
        self._d_model = d_model

        rand_idm = torch.eye(self.d_state).unsqueeze(0) + torch.randn(
            self.d_model, self.d_state, self.d_state
        ) / torch.sqrt(torch.tensor(self.d_state))
        self.A = nn.Parameter(rand_idm)
        self.proj = nn.Linear(self.d_state, 1)
        self.C = nn.Linear(self.d_state, self.d_model)
        self.D = nn.Linear(self.d_model, self.d_model)

        self.activation_fn = get_activation(activation, functional=True)

    def forward_slow(self, x: Tensor) -> Tensor:
        """Forward pass (slow version)."""
        Ax = einsum(x, self.A, "b l dm, dm ds dst -> b l ds dst")  # noqa: N806
        Ax = self.activation_fn(Ax)  # noqa: N806  (B, L, d_state, d_state)

        cum_prod = []
        for i in range(Ax.shape[0]):
            A_i = torch.unbind(Ax[i], dim=0)  # noqa: N806 [(d_state, d_state), ...] * L
            prod_list = list(accumulate(A_i, lambda x, y: x @ y))
            prod_list = [self.proj(p) for p in prod_list]
            A_i = torch.stack(prod_list, dim=0).squeeze()  # noqa: N806 (L, d_state)
            cum_prod.append(A_i)
        cum_prod = torch.stack(cum_prod, dim=0).squeeze()  # (B, L, d_state)

        return self.C(cum_prod) + self.D(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, L, d_model)
        """
        Ax = einsum(x, self.A, "b l dm, dm ds dst -> b l ds dst")  # noqa: N806
        Ax = self.activation_fn(Ax)  # noqa: N806  (B, L, d_state, d_state)

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
    def activation(self) -> str:  # noqa: D102
        return self._activation

    @property
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(  # noqa: D107
        self,
        activation: str,
        d_model: int,
        d_state: int,
        n_layers: int,
        n_vocab: int,
        dropout: float,
        bias: bool = True,
        prenorm: bool = False,
    ):
        super().__init__()

        self._activation = activation
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
            self.s4_layers.append(
                IDS4Block(
                    d_model=self.d_model,
                    d_state=self.d_state,
                    activation=self.activation,
                )
            )
            self.norms.append(nn.LayerNorm(self.d_model))
            self.dropout_layers.append(nn.Dropout(self.dropout))

        self.cl_head = nn.Linear(self.d_model, self.n_vocab, self.bias)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.embedding(x)  # (B, L, d_input) -> (B, L, d_model)

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

    def forward_slow(self, x: Tensor) -> Tensor:
        """Forward pass (slow version)."""
        x = self.embedding(x)  # (B, L, d_input) -> (B, L, d_model)
        # x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(
            self.s4_layers, self.norms, self.dropout_layers
        ):
            z = x
            if self.prenorm:
                z = norm(z)

            z = layer.forward_slow(z)
            z = dropout(z)
            x = x + z

            if not self.prenorm:
                x = norm(x)

        x = self.cl_head(x)

        return x
