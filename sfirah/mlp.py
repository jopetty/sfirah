"""Provides Multi-layer Perceptrion implementations."""


import logging
from copy import deepcopy

from torch import Tensor, nn

from .utils import get_activation

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """A multi-layer perceptron.

    A standard MLP which is _not_ channel-wise independent;
    Each sequence in a batch is flattened so that the MLP can
    compute non-local interactions between tokens. This makes it
    different from the FF block in a Transformer.
    """

    @property
    def d_model(self) -> int:  # noqa: D102
        return self._d_model

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def d_ff(self) -> int:  # noqa: D102
        return self._d_ff

    @property
    def dropout(self) -> float:  # noqa: D102
        return self._dropout

    @property
    def activation(self) -> str:  # noqa: D102
        return self._activation

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def seq_len(self) -> int:  # noqa: D102
        return self._seq_len

    @property
    def layer_norm_eps(self) -> float:  # noqa: D102
        return self._layer_norm_eps

    @property
    def weight_scale(self) -> float:  # noqa: D102
        return self._weight_scale

    @property
    def weight_sharing(self) -> bool:  # noqa: D102
        return self._weight_sharing

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation: str,
        n_layers: int,
        weight_sharing: bool,
        weight_scale: float,
        layer_norm_eps: float,
        bias: bool,
        seq_len: int,
    ):
        """Initialize the MLP.

        Args:
            d_model (int): The embedding/model dimension.
            d_ff (int): The feed-forward dimension.
            dropout (float): The dropout probability.
            activation (str): The activation function.
            n_layers (int): The number of layers.
            n_vocab (int): The number of vocabulary tokens.
            weight_sharing (bool): Whether to share weights across layers.
            weight_scale (float): How much initial weights are scaled by.
            layer_norm_eps (float): The layer norm epsilon.
            bias (bool): Whether to include bias parameters.
            seq_len (int): The sequence length.
        """
        super().__init__()

        self._d_model = d_model
        self._d_ff = d_ff
        self._dropout = dropout
        self._activation = activation
        self._n_layers = n_layers
        self._weight_sharing = weight_sharing
        self._weight_scale = weight_scale
        self._layer_norm_eps = layer_norm_eps
        self._bias = bias
        self._seq_len = seq_len

        ff_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * seq_len, d_ff, bias=bias),
            get_activation(activation, functional=False),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
            nn.LayerNorm(d_model, layer_norm_eps),
        )

        if self.weight_sharing:
            self.ff = nn.ModuleList([ff_layer] * n_layers)
        else:
            self.ff = nn.ModuleList([deepcopy(ff_layer) for _ in range(n_layers)])

        for _, p in self.named_parameters():
            p = p * weight_scale

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
        """
        if self.weight_sharing or len(self.ff) == 1:
            assert self.ff[0] == self.ff[-1], "Weights are not shared!"
        else:
            assert self.ff[0] != self.ff[-1], "Weights are shared!"

        x = self.embedding(x)
        for ff in self.ff:
            x = ff(x)

        return x


class MLPClassifier(MLP):
    """An MLP classifier with no embeddings."""

    @property
    def n_classes(self) -> int:  # noqa: D102
        return self._n_classes

    def __init__(self, n_classes: int, **kwargs):
        """Initialize the MLPClassifier."""
        super().__init__(**kwargs)

        self._n_classes = n_classes

        self.cl_head = nn.Linear(self.d_model, self.n_classes, bias=self.bias)

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
        """
        x = super().forward(x)
        x = self.cl_head(x)
        return x


class SequenceMLP(nn.Module):
    """A multi-layer perceptron.

    A standard MLP which is _not_ channel-wise independent;
    Each sequence in a batch is flattened so that the MLP can
    compute non-local interactions between tokens. This makes it
    different from the FF block in a Transformer.
    """

    @property
    def d_model(self) -> int:  # noqa: D102
        return self._d_model

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def d_ff(self) -> int:  # noqa: D102
        return self._d_ff

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
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def seq_len(self) -> int:  # noqa: D102
        return self._seq_len

    @property
    def layer_norm_eps(self) -> float:  # noqa: D102
        return self._layer_norm_eps

    @property
    def weight_scale(self) -> float:  # noqa: D102
        return self._weight_scale

    @property
    def weight_sharing(self) -> bool:  # noqa: D102
        return self._weight_sharing

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation: str,
        n_layers: int,
        n_vocab: int,
        weight_sharing: bool,
        weight_scale: float,
        layer_norm_eps: float,
        bias: bool,
        seq_len: int,
    ):
        """Initialize the MLP.

        Args:
            d_model (int): The embedding/model dimension.
            d_ff (int): The feed-forward dimension.
            dropout (float): The dropout probability.
            activation (str): The activation function.
            n_layers (int): The number of layers.
            n_vocab (int): The number of vocabulary tokens.
            weight_sharing (bool): Whether to share weights across layers.
            weight_scale (float): How much initial weights are scaled by.
            layer_norm_eps (float): The layer norm epsilon.
            bias (bool): Whether to include bias parameters.
            seq_len (int): The sequence length.
        """
        super().__init__()

        self._d_model = d_model
        self._d_ff = d_ff
        self._dropout = dropout
        self._activation = activation
        self._n_layers = n_layers
        self._n_vocab = n_vocab
        self._weight_sharing = weight_sharing
        self._weight_scale = weight_scale
        self._layer_norm_eps = layer_norm_eps
        self._bias = bias
        self._seq_len = seq_len

        self.embedding = nn.Embedding(n_vocab, d_model)

        ff_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * seq_len, d_ff, bias=bias),
            get_activation(activation, functional=False),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
            nn.LayerNorm(d_model, layer_norm_eps),
        )

        if self.weight_sharing:
            self.ff = nn.ModuleList([ff_layer] * n_layers)
        else:
            self.ff = nn.ModuleList([deepcopy(ff_layer) for _ in range(n_layers)])

        for _, p in self.named_parameters():
            p = p * weight_scale

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
        """
        if self.weight_sharing or len(self.ff) == 1:
            assert self.ff[0] == self.ff[-1], "Weights are not shared!"
        else:
            assert self.ff[0] != self.ff[-1], "Weights are shared!"

        x = self.embedding(x)
        for ff in self.ff:
            x = ff(x)

        return x


class MLPSequenceClassifier(SequenceMLP):
    """An MLP with a linear classifier head."""

    def __init__(self, **kwargs):
        """Initialize the MLPSequenceClassifier."""
        super().__init__(**kwargs)

        self.cl_head = nn.Linear(self.d_model, self.n_vocab, bias=self.bias)

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
        """
        x = super().forward(x)
        x = self.cl_head(x)
        return x
