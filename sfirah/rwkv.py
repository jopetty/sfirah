"""RWKV model."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from .transformers import SinusoidalPositionalEncoding


class RWKV_TimeMix_x051a(nn.Module):  # noqa: N801
    """RWKV TimeMix x051a."""

    @property
    def head_size(self) -> int:
        """The dimension of each head."""
        return self.d_embedding // self.n_heads

    @property
    def d_embedding(self) -> int:  # noqa: D102
        return self._d_embedding

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def n_heads(self) -> int:  # noqa: D102
        return self._n_heads

    def __init__(  # noqa: D107
        self,
        d_embedding: int,
        bias: bool,
        n_heads: int,
        dropout: float | None,
        n_layers: int,
        layer_id: int,
    ):
        super().__init__()

        self._d_embedding = d_embedding
        self._bias = bias
        self._n_heads = n_heads
        self._dropout = dropout
        self._n_layers = n_layers

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layers - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layers)
            ddd = torch.ones(1, 1, d_embedding)
            for i in range(d_embedding):
                ddd[0, 0, i] = i / d_embedding

            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)
            )
            self.time_maa_g = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)
            )

            decay_speed = torch.ones(n_heads)
            for h in range(n_heads):
                decay_speed[h] = -6 + 5 * (h / (n_heads - 1)) ** (
                    0.7 + 1.3 * ratio_0_to_1
                )
            self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))

            tmp = torch.zeros(n_heads)
            for h in range(n_heads):
                tmp[h] = ratio_0_to_1 * (1 - (h / (n_heads - 1)))
            self.time_faaaa = nn.Parameter(tmp.unsqueeze(-1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(d_embedding, d_embedding, bias=bias)
        self.key = nn.Linear(d_embedding, d_embedding, bias=bias)
        self.value = nn.Linear(d_embedding, d_embedding, bias=bias)
        self.gate = nn.Linear(d_embedding, d_embedding, bias=bias)

        self.output = nn.Linear(d_embedding, d_embedding, bias=bias)
        self.ln_x = nn.GroupNorm(n_heads, d_embedding, eps=(1e-5) * 64)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        # batch_size, sequence_length, d_embedding
        B, T, C = x.size()  # noqa: N806
        H, N = self.n_heads, self.head_size  # noqa: N806

        if T % 256 == 0:
            Q = 256  # noqa: N806
        elif T % 128 == 0:
            Q = 128  # noqa: N806
        else:
            Q = T  # noqa: N806
        assert T % Q == 0

        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        xg = x + xx * self.time_maa_g
        r = self.receptance(xr).view(B, T, H, N).transpose(1, 2)  # receptance
        k = self.key(xk).view(B, T, H, N).permute(0, 2, 3, 1)  # key
        v = self.value(xv).view(B, T, H, N).transpose(1, 2)  # value
        g = F.silu(self.gate(xg))  # extra gate

        w = torch.exp(-torch.exp(self.time_decay.float()))  # time_decay
        u = self.time_faaaa.float()  # time_first

        ws = w.pow(Q).view(1, H, 1, 1)

        ind = torch.arange(Q - 1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, Q).pow(ind)

        wk = w.view(1, H, 1, Q)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, Q))
        w = torch.tile(w, [Q])
        w = w[:, :-Q].view(-1, Q, 2 * Q - 1)
        w = w[:, :, Q - 1 :].view(1, H, Q, Q)

        w = w.to(dtype=r.dtype)  # the decay matrix
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        state = torch.zeros(B, H, N, N, device=r.device, dtype=r.dtype)  # state
        y = torch.empty(B, H, T, N, device=r.device, dtype=r.dtype)  # output

        for i in range(T // Q):  # the rwkv-x051a operator
            rr = r[:, :, i * Q : i * Q + Q, :]
            kk = k[:, :, :, i * Q : i * Q + Q]
            vv = v[:, :, i * Q : i * Q + Q, :]
            y[:, :, i * Q : i * Q + Q, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
            state = ws * state + (kk * wk) @ vv

        y = y.transpose(1, 2).contiguous().view(B * T, C)
        y = self.ln_x(y).view(B, T, C) * g

        # output projection
        y = self.dropout(self.output(y))
        return y


class RWKV_ChannelMix_x051a(nn.Module):  # noqa: N801
    """RWKV ChannelMix x051a."""

    @property
    def head_size(self) -> int:
        """The dimension of each head."""
        return self.d_embedding // self.n_heads

    @property
    def d_embedding(self) -> int:  # noqa: D102
        return self._d_embedding

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def n_heads(self) -> int:  # noqa: D102
        return self._n_heads

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def dropout(self) -> float | None:  # noqa: D102
        return self._dropout

    def __init__(  # noqa: D107
        self,
        d_embedding: int,
        bias: bool,
        n_heads: int,
        dropout: float | None,
        n_layers: int,
        layer_id: int,
    ):
        super().__init__()

        self._d_embedding = d_embedding
        self._bias = bias
        self._n_heads = n_heads
        self._dropout = dropout
        self._n_layers = n_layers

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layers)
            ddd = torch.ones(1, 1, d_embedding)
            for i in range(d_embedding):
                ddd[0, 0, i] = i / d_embedding
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(d_embedding, 3 * d_embedding, bias=bias)
        self.value = nn.Linear(3 * d_embedding, d_embedding, bias=bias)
        self.receptance = nn.Linear(d_embedding, d_embedding, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # noqa: D102
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        x = self.key(xk)
        x = torch.relu(x) ** 2
        x = self.value(x)
        x = torch.sigmoid(self.receptance(xr)) * x
        x = self.dropout(x)
        return x


class RWKVBlock(nn.Module):
    """RWKV block."""

    @property
    def head_size(self) -> int:
        """The dimension of each head."""
        return self.d_embedding // self.n_heads

    @property
    def d_embedding(self) -> int:  # noqa: D102
        return self._d_embedding

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def n_heads(self) -> int:  # noqa: D102
        return self._n_heads

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def dropout(self) -> float | None:  # noqa: D102
        return self._dropout

    @property
    def layer_id(self) -> int:  # noqa: D102
        return self._layer_id

    def __init__(  # noqa: D107
        self,
        d_embedding: int,
        bias: bool,
        n_heads: int,
        dropout: float | None,
        n_layers: int,
        layer_id: int,
    ):
        super().__init__()

        self._d_embedding = d_embedding
        self._bias = bias
        self._n_heads = n_heads
        self._dropout = dropout
        self._n_layers = n_layers
        self._layer_id = layer_id

        self.ln_1 = nn.LayerNorm(d_embedding, bias=bias)
        self.tmix = RWKV_TimeMix_x051a(
            d_embedding=d_embedding,
            bias=bias,
            n_heads=n_heads,
            dropout=dropout,
            n_layers=n_layers,
            layer_id=layer_id,
        )
        self.ln_2 = nn.LayerNorm(d_embedding, bias=bias)
        self.cmix = RWKV_ChannelMix_x051a(
            d_embedding=d_embedding,
            bias=bias,
            n_heads=n_heads,
            dropout=dropout,
            n_layers=n_layers,
            layer_id=layer_id,
        )

    def forward(self, x):  # noqa: D102
        x = x + self.tmix(self.ln_1(x))
        x = x + self.cmix(self.ln_2(x))
        return x


class RWKV(nn.Module):
    """RWKV model."""

    def __init__(  # noqa: D107
        self,
        bias: bool,
        d_model: int,
        dropout: float | None,
        n_heads: int,
        n_layers: int,
        n_vocab: int,
        weight_sharing: bool,
    ):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Embedding(n_vocab, d_model),
            SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout),
        )

        layer = RWKVBlock(
            d_embedding=d_model,
            bias=bias,
            n_heads=n_heads,
            dropout=dropout,
            n_layers=n_layers,
            layer_id=0,
        )

        if self.weight_sharing:
            self.blocks = nn.ModuleList([layer] * n_layers)
        else:
            self.blocks = nn.ModuleList([layer for _ in range(n_layers)])
