"""S4 models."""


import logging

from s4.models.s4.s4d import S4D
from torch import Tensor, nn

from .layers import IndexPool

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class S4TokenClassifier(nn.Module):
    """S4 Token Classifier."""

    @property
    def cl_dim(self) -> int:  # noqa: D102
        return self._cl_dim

    @property
    def cl_index(self) -> int:  # noqa: D102
        return self._cl_index

    @property
    def d_model(self) -> int:  # noqa: D102
        return self._d_model

    @property
    def d_state(self) -> int:  # noqa: D102
        return self._d_state

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self._n_vocab

    @property
    def dropout(self) -> float:  # noqa: D102
        return self._dropout

    @property
    def transposed(self) -> bool:  # noqa: D102
        return self._transposed

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def prenorm(self) -> bool:  # noqa: D102
        return self._prenorm

    @property
    def num_parameters(self) -> int:  # noqa: D102
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(  # noqa: D107
        self,
        d_model: int,
        d_state: int,
        dropout: float,
        n_layers: int,
        n_vocab: int,
        lr: float,
        cl_index: int,
        cl_dim: int,
        transposed: bool = True,
        prenorm: bool = False,
    ):
        super().__init__()
        self._d_model = d_model
        self._d_state = d_state
        self._dropout = dropout
        self._n_vocab = n_vocab
        self._n_layers = n_layers
        self._prenorm = prenorm
        self._transposed = transposed
        self._cl_dim = cl_dim
        self._cl_index = cl_index

        self.embedding = nn.Embedding(n_vocab, d_model)

        self.s4d_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.s4d_layers.append(
                S4D(d_model=d_model, dropout=dropout, transposed=transposed, lr=lr)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropout_layers.append(nn.Dropout(dropout))

        self.cl_head = nn.Sequential(
            IndexPool(dim=cl_dim, index=cl_index),
            nn.Linear(
                self.d_model,
                self.n_vocab,
                self.bias,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.embedding(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)

        for layer, norm, dropout in zip(
            self.s4d_layers, self.norms, self.dropout_layers
        ):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)
            z = dropout(z)
            x = x + z

            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)
        x = self.cl_head(x)

        return x
