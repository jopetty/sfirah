"""Recurrent neural netorks."""

import logging

from torch import Tensor, nn

from .layers import IndexPool

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SRN(nn.Module):
    """A simple recurrent network."""

    @property
    def d_embedding(self) -> int:  # noqa: D102
        return self._d_embedding

    @property
    def d_hidden(self) -> int:  # noqa: D102
        return self._d_hidden

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def dropout(self) -> float:  # noqa: D102
        return self._dropout

    @property
    def activation(self) -> str:  # noqa: D102
        return self._activation

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self._n_vocab

    @property
    def batch_first(self) -> bool:  # noqa: D102
        return self._batch_first

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def weight_scale(self) -> float:  # noqa: D102
        return self._weight_scale

    @property
    def num_parameters(self) -> int:  # noqa: D102
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(
        self,
        activation: str,
        batch_first: bool,
        bias: bool,
        dropout: float | None,
        d_embedding: int,
        d_hidden: int,
        n_layers: int,
        n_vocab: int,
        weight_scale: float,
    ):
        """Initialize an SRN.

        Args:
            activation (str): The activation function.
            batch_first (bool): Whether the batch is first.
            bias (bool): Whether to use bias.
            dropout (float | None): The dropout rate.
            d_embedding (int): The embedding dimension.
            d_hidden (int): The hidden dimension.
            n_layers (int): The number of layers.
            n_vocab (int): The number of vocabulary.
            weight_scale (float): The weight scale.

        """
        self._activation = activation
        self._batch_first = batch_first
        self._bias = bias
        self._dropout = dropout
        self._d_embedding = d_embedding
        self._d_hidden = d_hidden
        self._n_layers = n_layers
        self._n_vocab = n_vocab
        self._weight_scale = weight_scale

        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=d_embedding)
        self.rnn = nn.RNN(
            input_size=d_embedding,
            hidden_size=d_hidden,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            bidirectional=False,
        )

        # TODO: weight sharing

        for _, p in self.named_parameters():
            p = p * weight_scale

    def forward(self, x: Tensor, h: Tensor | None = None) -> (Tensor, Tensor):
        """Forward pass."""
        x = self.embedding(x)

        if h is None:
            h = x.new_zeros(self.n_layers, x.size(0), self.d_hidden)

        x, h = self.rnn(input=x, hx=h)
        return x, h


class SRNSequenceClassifier(SRN):
    """An SRN sequence-level classifier."""

    def __init__(self, cl_dim: int, cl_index: int, **kwargs: dict):  # noqa: D107
        super().__init__(**kwargs)
        self.cl_head = nn.Sequential(
            IndexPool(cl_dim, cl_index),
            nn.Linear(self.d_hidden, self.n_vocab, self.bias),
        )

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        """Perform the forward pass."""
        x, _ = super().forward(x=x, h=h)
        x = self.cl_head(x)
        return x


class SRNTokenClassifier(SRN):
    """An SRN token-level classifier."""

    def __init__(self, **kwargs):  # noqa: D107
        super().__init__(**kwargs)

        self.cl_head = nn.Linear(self.d_hidden, self.n_vocab, self.bias)

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        """Perform the forward pass."""
        x, _ = super().forward(x=x, h=h)
        x = self.cl_head(x)
        return x


class GRU(nn.Module):
    """A GRU."""

    @property
    def d_embedding(self) -> int:  # noqa: D102
        return self._d_embedding

    @property
    def d_hidden(self) -> int:  # noqa: D102
        return self._d_hidden

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def dropout(self) -> float:  # noqa: D102
        return self._dropout

    @property
    def activation(self) -> str:  # noqa: D102
        return self._activation

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self._n_vocab

    @property
    def batch_first(self) -> bool:  # noqa: D102
        return self._batch_first

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def weight_scale(self) -> float:  # noqa: D102
        return self._weight_scale

    @property
    def num_parameters(self) -> int:  # noqa: D102
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(
        self,
        activation: str,
        batch_first: bool,
        bias: bool,
        dropout: float | None,
        d_embedding: int,
        d_hidden: int,
        n_layers: int,
        n_vocab: int,
        weight_scale: float,
    ):
        """Initialize a GRU.

        Args:
            activation (str): The activation function.
            batch_first (bool): Whether the batch is first.
            bias (bool): Whether to use bias.
            dropout (float | None): The dropout rate.
            d_embedding (int): The embedding dimension.
            d_hidden (int): The hidden dimension.
            n_layers (int): The number of layers.
            n_vocab (int): The number of vocabulary.
            weight_scale (float): The weight scale.

        """
        self._activation = activation
        self._batch_first = batch_first
        self._bias = bias
        self._dropout = dropout
        self._d_embedding = d_embedding
        self._d_hidden = d_hidden
        self._n_layers = n_layers
        self._n_vocab = n_vocab
        self._weight_scale = weight_scale

        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=d_embedding)
        self.rnn = nn.GRU(
            input_size=d_embedding,
            hidden_size=d_hidden,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            bidirectional=False,
        )

        # TODO: weight sharing

        for _, p in self.named_parameters():
            p = p * weight_scale

    def forward(self, x: Tensor, h: Tensor | None = None) -> (Tensor, Tensor):
        """Forward pass."""
        x = self.embedding(x)

        if h is None:
            h = x.new_zeros(self.n_layers, x.size(0), self.d_hidden)

        x, h = self.rnn(input=x, hx=h)
        return x, h


class GRUSequenceClassifier(GRU):
    """A GRU sequence-level classifier."""

    def __init__(self, cl_dim: int, cl_index: int, **kwargs: dict):  # noqa: D107
        super().__init__(**kwargs)
        self.cl_head = nn.Sequential(
            IndexPool(cl_dim, cl_index),
            nn.Linear(self.d_hidden, self.n_vocab, self.bias),
        )

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        """Perform the forward pass."""
        x, _ = super().forward(x=x, h=h)
        x = self.cl_head(x)
        return x


class GRUTokenClassifier(GRU):
    """A GRU token-level classifier."""

    def __init__(self, **kwargs):  # noqa: D107
        super().__init__(**kwargs)

        self.cl_head = nn.Linear(self.d_hidden, self.n_vocab, self.bias)

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(self, x: Tensor, h: Tensor | None = None) -> Tensor:
        """Perform the forward pass."""
        x, _ = super().forward(x=x, h=h)
        x = self.cl_head(x)
        return x


class LSTM(nn.Module):
    """An LSTM."""

    @property
    def d_embedding(self) -> int:  # noqa: D102
        return self._d_embedding

    @property
    def d_hidden(self) -> int:  # noqa: D102
        return self._d_hidden

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def dropout(self) -> float:  # noqa: D102
        return self._dropout

    @property
    def activation(self) -> str:  # noqa: D102
        return self._activation

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self._n_vocab

    @property
    def batch_first(self) -> bool:  # noqa: D102
        return self._batch_first

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def weight_scale(self) -> float:  # noqa: D102
        return self._weight_scale

    @property
    def num_parameters(self) -> int:  # noqa: D102
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(
        self,
        activation: str,
        batch_first: bool,
        bias: bool,
        dropout: float | None,
        d_embedding: int,
        d_hidden: int,
        n_layers: int,
        n_vocab: int,
        weight_scale: float,
    ):
        """Initialize an LSTM.

        Args:
            activation (str): The activation function.
            batch_first (bool): Whether the batch is first.
            bias (bool): Whether to use bias.
            dropout (float | None): The dropout rate.
            d_embedding (int): The embedding dimension.
            d_hidden (int): The hidden dimension.
            n_layers (int): The number of layers.
            n_vocab (int): The number of vocabulary.
            weight_scale (float): The weight scale.

        """
        self._activation = activation
        self._batch_first = batch_first
        self._bias = bias
        self._dropout = dropout
        self._d_embedding = d_embedding
        self._d_hidden = d_hidden
        self._n_layers = n_layers
        self._n_vocab = n_vocab
        self._weight_scale = weight_scale

        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=d_embedding)
        self.lstm = nn.LSTM(
            input_size=d_embedding,
            hidden_size=d_hidden,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            bidirectional=False,
        )

        # TODO: add option for proj_size?
        # TODO: weight sharing

        for _, p in self.named_parameters():
            p = p * weight_scale

    def forward(
        self, x: Tensor, h: Tensor | None = None, c: Tensor | None = None
    ) -> (Tensor, (Tensor, Tensor)):
        """Forward pass."""
        x = self.embedding(x)

        if h is None:
            h = x.new_zeros(self.n_layers, x.size(0), self.d_hidden)

        if c is None:
            c = x.new_zeros(self.n_layers, x.size(0), self.d_hidden)

        x, hc = self.lstm(input=x, hx=(h, c))
        return x, hc


class LSTMSequenceClassifier(LSTM):
    """An LSTM sequence-level classifier."""

    def __init__(self, cl_dim: int, cl_index: int, **kwargs: dict):  # noqa: D107
        super().__init__(**kwargs)
        self.cl_head = nn.Sequential(
            IndexPool(cl_dim, cl_index),
            nn.Linear(self.d_hidden, self.n_vocab, self.bias),
        )

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(
        self, x: Tensor, h: Tensor | None = None, c: Tensor | None = None
    ) -> Tensor:
        """Perform the forward pass."""
        x, _ = super().forward(x=x, hx=(h, c))
        x = self.cl_head(x)
        return x


class LSTMTokenClassifier(LSTM):
    """An LSTM token-level classifier."""

    def __init__(self, **kwargs):  # noqa: D107
        super().__init__(**kwargs)

        self.cl_head = nn.Linear(self.d_hidden, self.n_vocab, self.bias)

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(
        self, x: Tensor, h: Tensor | None = None, c: Tensor | None = None
    ) -> Tensor:
        """Perform the forward pass."""
        x, _ = super().forward(x=x, hc=(h, c))
        x = self.cl_head(x)
        return x
