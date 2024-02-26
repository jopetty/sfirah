"""State-space models."""


import logging
from functools import partial
from typing import Any

import torch
from einops import rearrange
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.modules.mamba_simple import Block as MambaBlock
from mamba_ssm.modules.mamba_simple import Mamba as MambaBase
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

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


class IDMambaMambaBase(MambaBase):
    """Mamba but A_bar is now input dependent."""

    def step(self, hidden_states, conv_state, ssm_state):  # noqa: D102
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)  # noqa: N806, E501
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  #  noqa: N806 (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))  # noqa: N806
            dB = torch.einsum("bd,bn->bdn", dt, B)  # noqa: N806
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


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
