import logging

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from .layers import IndexPool
from .utils import get_activation

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):

    @property
    def weight_sharing(self) -> bool:
        return self._weight_sharing

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        activation: str,
        n_layers: int,
        norm_first: bool,
        layer_norm_eps: float,
        batch_first: bool,
        weight_sharing: bool,
        n_vocab: int,
        weight_scale: int,
        bias: bool,
    ):
        super().__init__()

        self._weight_sharing = weight_sharing
        self._n_layers = n_layers

        self.embedding = nn.Sequential(
            nn.Embedding(n_vocab, d_model),
            SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=get_activation(activation, functional=True),
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        if self.weight_sharing:
            self.transformer.layers = nn.ModuleList([layer] * n_layers)

        for _, p in self.named_parameters():
            p = p * weight_scale

    def forward(
        self,
        x: torch.Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        return x


class EncoderClassifier(Transformer):
    def __init__(self, cl_dim: int, cl_index: int, **kwargs: dict):
        super().__init__(**kwargs)

        self.cl_head = nn.Sequential(
            IndexPool(dim=cl_dim, index=cl_index),
            nn.Linear(
                kwargs["d_model"],
                kwargs["n_classes"],
                bias=kwargs["bias"],
            ),
        )

        for _, p in self.cl_head.named_parameters():
            p = p * kwargs["weight_scale"]

    def forward(
        self,
        x: torch.Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = super().forward(
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        x = self.cl_head(x)
        return x


class CausalDecoder(Transformer):
    @property
    def context_size(self) -> int:
        return self._context_size

    def __init__(self, context_size: int, **kwargs):
        super().__init__(**kwargs)
        self._context_size = context_size
        self.lm_head = nn.Linear(
            kwargs["d_model"],
            kwargs["n_vocab"],
            bias=kwargs["bias"],
        )

        for _, p in self.lm_head.named_parameters():
            p = p * kwargs["weight_scale"]

    def forward(
        self,
        x: torch.Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        x = super().forward(
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        x = self.lm_head(x)

        # logits are of shape (batch_size, seq_len, n_vocab), but
        # F.cross_entropy expects (batch_size, n_vocab, seq_len)
        x = x.transpose(-1, -2)
        return x

    @torch.no_grad()
    def generate(
        self,
        context: Tensor,
        max_length: int = 2048,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        seq_len = context.shape[1]
        max_new_tokens = max_length - seq_len

        assert max_new_tokens > 0, "Context is longer than max_length"

        for _ in range(max_new_tokens):
            if seq_len > self.context_size:
                context = context[:, -self.context_size :]

            logits = self.forward(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k=min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=1)
            tok_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, tok_next], dim=-1)

        return context
