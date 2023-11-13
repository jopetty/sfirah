"""Transformer models."""


import logging
from typing import Any

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
    """Sinusoidal positional encodings.

    This module is copied from the PyTorch implementation, which for
    some reason is not included as an official module.

    See: https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize the SinusoidalPositionalEncoding module.

        Args:
            d_model (int): The model/embedding dimension.
            dropout (float): The dropout probability. Defaults to 0.1.
            max_len (int): The maximum length of the sequence. Defaults to 5000.
        """
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

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    """A Transformer (encoder).

    Provides a standard implementation of a transformer. This module is built
    around PyTorch's TransformerEncoderLayer, but this is somewhat misleading---
    this can be used as either an encoder or a decoder, depending on the type
    of attention mask and the head attached.
    """

    @property
    def d_model(self) -> int:  # noqa: D102
        return self._d_model

    @property
    def n_heads(self) -> int:  # noqa: D102
        return self._n_heads

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
    def n_layers(self) -> int:  # noqa: D102
        return self._n_layers

    @property
    def norm_first(self) -> bool:  # noqa: D102
        return self._norm_first

    @property
    def layer_norm_eps(self) -> float:  # noqa: D102
        return self._layer_norm_eps

    @property
    def batch_first(self) -> bool:  # noqa: D102
        return self._batch_first

    @property
    def weight_sharing(self) -> bool:  # noqa: D102
        return self._weight_sharing

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self._n_vocab

    @property
    def weight_scale(self) -> int:  # noqa: D102
        return self._weight_scale

    @property
    def bias(self) -> bool:  # noqa: D102
        return self._bias

    @property
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
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
        """Initialize the Transformer module.

        Args:
            d_model (int): The model/embedding dimension.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimension of the feed-forward layer.
            dropout (float): The dropout probability.
            activation (str): The activation function.
            n_layers (int): The number of layers.
            norm_first (bool): Whether to add layer-norm before attention.
            layer_norm_eps (float): The layer-norm epsilon.
            batch_first (bool): Whether the batch dimension is first.
            weight_sharing (bool): Whether to share weights across layers.
            n_vocab (int): The number of vocabulary items.
            weight_scale (int): How much initial weights are scaled by.
            bias (bool): Whether to include bias parameters.
        """
        super().__init__()

        self._d_model = d_model
        self._n_heads = n_heads
        self._d_ff = d_ff
        self._dropout = dropout
        self._activation = activation
        self._n_layers = n_layers
        self._norm_first = norm_first
        self._layer_norm_eps = layer_norm_eps
        self._batch_first = batch_first
        self._weight_sharing = weight_sharing
        self._n_vocab = n_vocab
        self._weight_scale = weight_scale
        self._bias = bias

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

        # TODO: Does weight_sharing break this? If we share weights are we
        #       scaling the layers multiple times? Check to make sure this
        #       isn't happening!
        for _, p in self.named_parameters():
            p = p * weight_scale

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor | None): The attention mask.
            src_key_padding_mask (Tensor | None): The source key padding mask.
            is_causal (bool): Whether the attention is causal.
        """
        x = self.embedding(x)
        x = self.transformer(
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        return x


class EncoderSequenceClassifier(Transformer):
    """A Transformer encoder with a linear classifier head & sequence pooling.

    This is a standard Transformer encoder with a linear classifier head
    which builds sentence representations by index pooling to a particular
    token in the input (e.g., the CLS token).

    This model us useful for classifying sequences; e.g., for sentiment
    classification.
    """

    def __init__(self, cl_dim: int, cl_index: int, **kwargs: dict):
        """Initialize the EncoderSequenceClassifier.

        Args:
            cl_dim (int): The dimension to pool.
            cl_index (int): The index to pool.
            **kwargs (dict): Additional keyword arguments.
        """
        super().__init__(**kwargs)

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

    def forward(
        self,
        x: torch.Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor | None): The attention mask.
            src_key_padding_mask (Tensor | None): The source key padding mask.
            is_causal (bool): Whether the attention is causal.
        """
        x = super().forward(
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        x = self.cl_head(x)
        return x


class EncoderTokenClassifier(Transformer):
    """A Transformer encoder with a linear classifier head.

    This is a standard Transformer encoder with a linear classifier head
    which computes a per-token classification. No sequence pooling is done,
    so this model us suitable for token-level classification tasks, like
    POS-tagging.
    """

    def __init__(self, **kwargs):  # noqa: D107
        super().__init__(**kwargs)

        self.cl_head = nn.Linear(
            self.d_model,
            self.n_vocab,
            self.bias,
        )

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor | None): The attention mask.
            src_key_padding_mask (Tensor | None): The source key padding mask.
            is_causal (bool): Whether the attention is causal.
        """
        x = super().forward(
            x, mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal
        )
        x = self.cl_head(x)
        return x


class CausalDecoder(Transformer):
    """A Transformer with a causal-attention mask.

    This implements a GPT-style decoder-only model, suitable for autoregressive
    tasks.
    """

    @property
    def context_size(self) -> int:  # noqa: D102
        return self._context_size

    def __init__(self, context_size: int, **kwargs):
        """Initialize the CausalDecoder.

        Args:
            context_size (int): The size of the context window.
            **kwargs (dict): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._context_size = context_size
        self.lm_head = nn.Linear(
            self.d_model,
            self.n_vocab,
            bias=self.bias,
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
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor | None): The attention mask.
            src_key_padding_mask (Tensor | None): The source key padding mask.
            is_causal (bool): Whether the attention is causal.
        """
        x = super().forward(
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        context: Tensor,
        tokenizer: Any | None = None,
        max_length: int = 2048,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        stream: bool = False,
    ):
        r"""Generate a sequence of tokens conditioned on some context.

        Args:
            context (Tensor): The context/model "inputs".
            tokenizer: The tokenizer to use for decoding.
            max_length (int): The maximum length of the generated sequence.
            temperature (float): Controls how random the outputs are (0, \infty).
            top_k (int | None): The top-k sampling. Defaults to None.
            eos_token_id (int | None): The end-of-sequence token ID. Defaults to None.
            stream (bool): Whether to stream the output. Defaults to False.
        """
        seq_len = context.shape[1]
        max_new_tokens = max_length - seq_len

        assert (tokenizer is None) != (
            eos_token_id is None
        ), "You must provide either a tokenizer or an EOS token ID."

        if stream:
            assert tokenizer is not None, "You must provide a tokenizer to stream."
            print("streaming!")

        if tokenizer is not None:
            eos_token_id = tokenizer.eos_token_id

        assert max_new_tokens > 0, "Context is longer than max_length"
        assert temperature >= 0, "Temperature must be non-negative"

        # print(f"Temperature: {temperature}")
        if temperature == 0.0:
            logger.info("Generating with greedy decoding")
        else:
            logger.info(
                f"Generating with top-{top_k} sampling with temperature {temperature}"
            )

        for t in range(max_new_tokens):
            if seq_len > self.context_size:
                context = context[:, -self.context_size :]

            mask = torch.nn.Transformer.generate_square_subsequent_mask(
                context.shape[1], device=context.device
            )

            logits = self.forward(context, mask=mask)

            if temperature == 0.0:
                tok_next = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            else:
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k=min(top_k, logits.shape[-1]))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=1)
                tok_next = torch.multinomial(probs, num_samples=1)

            context = torch.cat([context, tok_next], dim=-1)
            if stream and tokenizer is not None:
                logger.info(tokenizer.decode(tok_next.squeeze()))

            if eos_token_id is not None and tok_next.item() == eos_token_id:
                logger.info(
                    f"EOS token generated. Stopping generation after {t+1} tokens."
                )
                break

        return context
