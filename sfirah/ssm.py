"""State-space models."""


import logging
from functools import partial
from typing import Any

from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.modules.mamba_simple import Block as MambaBlock
from mamba_ssm.modules.mamba_simple import Mamba as MambaBase
from s4.models.s4.s4 import S4Block
from torch import Tensor, nn

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .layers import IndexPool

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class S4(nn.Module):
    """An S4 model.

    Replicates the reference implementation from github.com/state-spaces/s4/example.py
    """

    @property
    def d_model(self) -> int:  # noqa: D102
        return self._d_model

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self._n_vocab

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def dropout(self) -> float:  # noqa: D102
        return self._dropout

    @property
    def norm_first(self) -> bool:  # noqa: D102
        return self._norm_first

    @property
    def lr(self) -> bool:  # noqa: D102
        return self._lr

    @property
    def num_parameters(self) -> int:  # noqa: D102
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(
        self,
        d_model: int,
        n_vocab: int,
        n_layers: int,
        dropout: float,
        norm_first: bool,
        lr: float | None = None,
    ):
        """Initialize an S4 module.

        Args:
            d_model (int): The model dimension.
            n_vocab (int): The vocabulary size.
            n_layers (int): The number of layers.
            dropout (float): The dropout rate.
            norm_first (bool): Whether to apply normalization before or after the mixer.
            lr (float, optional): The learning rate. Defaults to None.
        """
        self._d_model = d_model
        self._n_vocab = n_vocab
        self._n_layers = n_layers
        self._dropout = dropout
        self._norm_first = norm_first
        self._lr = lr

        self.embedding = nn.Embedding(n_vocab, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drouputs = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4Block(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.drouputs.append(nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass."""
        x = self.embedding(x)
        x = x.transpose(-1, -2)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.drouputs):
            z = x
            if self.norm_first:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            z, _ = layer(z)
            z = dropout(z)
            x = z + x

            if not self.norm_first:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)
        return x


class S4SequenceClassifier(S4):
    """An S4 model with a sequence classification head."""

    @property
    def cl_dim(self) -> int:  # noqa: D102
        return self._cl_dim

    @property
    def cl_index(self) -> int:  # noqa: D102
        return self._cl_index

    def __init__(self, cl_dim: int, cl_index: int, **kwargs: dict):  # noqa: D107
        super().__init__(**kwargs)

        self._cl_dim = cl_dim
        self._cl_index = cl_index

        self.cl_head = nn.Sequential(
            IndexPool(dim=cl_dim, index=cl_index),
            nn.Linear(
                self.d_model,
                self.n_vocab,
                self.bias,
            ),
        )

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass."""
        x = super().forward(x)
        x = self.cl_head(x)
        return x


class S4TokenClassifier(S4):
    """An S4 model with a token classification head."""

    def __init__(self, **kwargs):  # noqa: D107
        super().__init__(**kwargs)
        self.cl_head = nn.Linear(self.d_model, self.n_vocab, self.bias)

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass."""
        x = super().forward(x)
        x = self.cl_head(x)
        return x


class Mamba(nn.Module):
    """A Mamba model."""

    @property
    def d_model(self) -> int:  # noqa: D102
        return self._d_model

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def rms_norm(self) -> bool:  # noqa: D102
        return self._rms_norm

    @property
    def layer_norm_eps(self) -> float:  # noqa: D102
        return self._layer_norm_eps

    @property
    def fused_add_norm(self) -> bool:  # noqa: D102
        return self._fused_add_norm

    @property
    def residual_in_fp32(self) -> bool:  # noqa: D102
        return self._residual_in_fp32

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self._n_vocab

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def dropout(self) -> bool:  # noqa: D102
        return self._dropout

    @property
    def weight_scale(self) -> float:  # noqa: D102
        return self._weight_scale

    @property
    def num_parameters(self) -> int:  # noqa: D102
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_vocab: int,
        rms_norm: bool = False,
        layer_norm_eps: float = 1e-5,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        initializer_cfg: dict = None,
        bias: bool = False,
        dropout: float = 0.1,
        weight_scale: float = 1.0,
    ):
        """Initialize a bare Mamba module."""
        super().__init__()

        self._d_model = d_model
        self._n_layers = n_layers
        self._rms_norm = rms_norm
        self._layer_norm_eps = layer_norm_eps
        self._fused_add_norm = fused_add_norm
        self._residual_in_fp32 = residual_in_fp32
        self._n_vocab = n_vocab
        self._bias = bias
        self._dropout = dropout
        self._weight_scale = weight_scale

        self.embedding = nn.Embedding(n_vocab, d_model)

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                Mamba.create_block(
                    d_model=d_model,
                    ssm_cfg=None,
                    norm_epsilon=layer_norm_eps,
                    residual_in_fp32=residual_in_fp32,
                    rms_norm=rms_norm,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
                for i in range(n_layers)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=layer_norm_eps
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        for _, p in self.named_parameters():
            p = p * weight_scale

    def forward(self, x: Tensor, inference_params=None) -> Tensor:
        """Perform the forward pass."""
        x = self.embedding(x)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual, inference_params=inference_params)
        if not self.fused_add_norm:
            residual = (x + residual) if residual is not None else x
            x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            x = fused_add_norm_fn(
                x,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return x

    @staticmethod
    def create_block(
        d_model: int,
        ssm_cfg: dict[str, Any] = None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
    ):
        """Create a Mamba block."""
        if ssm_cfg is None:
            ssm_cfg = {}
        mixer_cls = partial(MambaBase, layer_idx=layer_idx, **ssm_cfg)
        norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
        block = MambaBlock(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        block.layer_idx = layer_idx
        return block


class MambaSequenceClassifier(Mamba):
    """A Mamba model with a sequence classification head."""

    @property
    def cl_dim(self) -> int:  # noqa: D102
        return self._cl_dim

    @property
    def cl_index(self) -> int:  # noqa: D102
        return self._cl_index

    def __init__(self, cl_dim: int, cl_index: int, **kwargs: dict):  # noqa: D107
        super().__init__(**kwargs)

        self._cl_dim = cl_dim
        self._cl_index = cl_index

        self.cl_head = nn.Sequential(
            IndexPool(dim=cl_dim, index=cl_index),
            nn.Linear(
                self.d_model,
                self.n_vocab,
                self.bias,
            ),
        )

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass."""
        x = super().forward(x)
        x = self.cl_head(x)
        return x


class MambaTokenClassifier(Mamba):
    """A Mamba model with a token classification head."""

    def __init__(self, **kwargs):  # noqa: D107
        super().__init__(**kwargs)
        self.cl_head = nn.Linear(self.d_model, self.n_vocab, self.bias)

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass."""
        x = super().forward(x)
        x = self.cl_head(x)
        return x
