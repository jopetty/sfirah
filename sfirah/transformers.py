"""Transformer models."""

import inspect
import logging
import math
from dataclasses import dataclass
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

    def __init__(self, cl_dim: int, cl_index: int | None, **kwargs: dict):
        """Initialize the EncoderSequenceClassifier.

        Args:
            cl_dim (int): The dimension to pool.
            cl_index (int | None): The index to pool.
            **kwargs (dict): Additional keyword arguments.
        """
        super().__init__(**kwargs)

        # self.cl_head = nn.Sequential(
        #     IndexPool(dim=cl_dim, index=cl_index),
        #     nn.Linear(
        #         self.d_model,
        #         self.n_vocab,
        #         self.bias,
        #     ),
        # )

        self.cl_head = nn.ModuleDict(
            {
                "pool": IndexPool(dim=cl_dim, index=cl_index),
                "linear": nn.Linear(
                    self.d_model,
                    self.n_vocab,
                    self.bias,
                ),
            }
        )

        for _, p in self.cl_head.named_parameters():
            p = p * self.weight_scale

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
        index: Tensor | None = None,
    ) -> torch.Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor | None): The attention mask.
            src_key_padding_mask (Tensor | None): The source key padding mask.
            is_causal (bool): Whether the attention is causal.
            index (Tensor | None): The index to pool.
        """
        x = super().forward(
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        x = self.cl_head["pool"](x, index=index)
        x = self.cl_head["linear"](x)
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


class GenerativeDecoder:
    """Wrapper class to provide a common generate() method."""

    @torch.no_grad()
    def generate(
        self,
        context: Tensor,
        tokenizer: Any | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
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
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): Controls how random the outputs are (0, \infty).
            top_k (int | None): The top-k sampling. Defaults to None.
            eos_token_id (int | None): The end-of-sequence token ID. Defaults to None.
            stream (bool): Whether to stream the output. Defaults to False.
        """
        seq_len = context.shape[1]
        assert (max_new_tokens is not None) != (
            max_length is not None
        ), "You must provide either `max_length` or `max_new_tokens`."
        if max_length is not None:
            max_new_tokens = max_length - seq_len
        assert max_new_tokens > 0, "Context is longer than max_length"

        assert (tokenizer is None) != (
            eos_token_id is None
        ), "You must provide either a tokenizer or an EOS token ID."

        if stream:
            assert tokenizer is not None, "You must provide a tokenizer to stream."
            logger.info("Streaming outputs.")

        if tokenizer is not None:
            eos_token_id = tokenizer.eos_token_id

        assert temperature >= 0, "Temperature must be non-negative"

        if temperature == 0.0:
            logger.info("Generating with greedy decoding")
        else:
            logger.info(
                f"Generating with top-{top_k} sampling with temperature {temperature}"
            )

        for t in range(max_new_tokens):
            if seq_len > self.block_size:
                context = context[:, -self.block_size :]

            logits, _ = self.forward(context)

            if temperature == 0.0:
                tok_next = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            else:
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                tok_next = torch.multinomial(probs, num_samples=1)

            context = torch.cat([context, tok_next], dim=-1)
            if stream and tokenizer is not None:
                next_token = tokenizer.decode(tok_next.squeeze())
                print(next_token, end=" ", flush=True)

            if eos_token_id is not None and tok_next.item() == eos_token_id:
                logger.info(
                    f"EOS token generated. Stopping generation after {t+1} tokens."
                )
                break

        return context

    @torch.no_grad()
    def top_k_completions(  # noqa: D102
        self, context: Tensor, k: int, tokenizer: Any
    ):
        seq_len = context.shape[1]
        if seq_len > self.block_size:
            context = context[:, -self.block_size :]

        logits, _ = self.forward(context)
        probs = F.softmax(logits, dim=-1).squeeze()
        torch.topk(logits, k=min(k, logits.shape[-1]))

        vals, indices = torch.topk(logits, k=min(k, logits.shape[-1]))
        vals = vals.squeeze()
        indices = indices.squeeze().tolist()

        completions = {}

        for i in range(k):
            idx = indices[i]
            completions[tokenizer.decode(idx)] = probs[idx].item()

        return completions


class CausalDecoder(Transformer, GenerativeDecoder):
    """A Transformer with a causal-attention mask.

    This implements a GPT-style decoder-only model, suitable for autoregressive
    tasks.
    """

    @property
    def block_size(self) -> int:  # noqa: D102
        return self._block_size

    def __init__(self, block_size: int, tie_embeddings: bool = True, **kwargs):
        """Initialize the CausalDecoder.

        Args:
            block_size (int): The size of the context window.
            tie_embeddings (bool): Whether to tie the embedding weights.
                Defaults to True.
            **kwargs (dict): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._block_size = block_size
        self.lm_head = nn.Linear(
            self.d_model,
            self.n_vocab,
            bias=self.bias,
        )

        # Reduces size of model file; for large vocabs, this can be significant.
        # Also means embedding get stronger gradients.
        if tie_embeddings:
            self.lm_head.weight = self.embedding[0].weight

        for _, p in self.lm_head.named_parameters():
            p = p * kwargs["weight_scale"]

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> Tensor:
        """Perform the forward pass.

        Args:
            idx (Tensor): The input tensor.
            targets (Tensor | None): The target tensor, if any targets are present.
        """
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            idx.shape[1], device=idx.device
        )

        x = super().forward(
            idx,
            mask=causal_mask,
            src_key_padding_mask=None,
            is_causal=True,
        )

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on
            # the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


# Mark: NanoGPT (karpathy)


@dataclass
class GPTConfig:
    """GPT Config object."""

    block_size: int = 1024
    vocab_size: int = (
        50304
    )  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2.
    # False: a bit better and faster


class GPTLayerNorm(nn.Module):
    """LayerNorm with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, ndim, bias):  # noqa: D107
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):  # noqa: D102
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class GPTCausalSelfAttention(nn.Module):
    """Causal self-attention."""

    def __init__(self, config: GPTConfig):  # noqa: D107
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left
            # in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):  # noqa: D102
        (
            B,  # noqa: N806
            T,  # noqa: N806
            C,  # noqa: N806
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to
        # be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend:
        #   (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class GPTMLP(nn.Module):
    """MLP for GPT."""

    def __init__(self, config):  # noqa: D107
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):  # noqa: D102
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPTBlock(nn.Module):
    """GPT Block."""

    def __init__(self, config):  # noqa: D107
        super().__init__()
        self.ln_1 = GPTLayerNorm(config.n_embd, bias=config.bias)
        self.attn = GPTCausalSelfAttention(config)
        self.ln_2 = GPTLayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GPTMLP(config)

    def forward(self, x):  # noqa: D102
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module, GenerativeDecoder):
    """karpathy's NanoGPT with a Sfirah interface."""

    @property
    def block_size(self) -> int:  # noqa: D102
        return self.config.block_size

    @property
    def d_model(self) -> int:  # noqa: D102
        return self.config.n_embd

    @property
    def n_heads(self) -> int:  # noqa: D102
        return self.config.n_head

    @property
    def d_ff(self) -> int:  # noqa: D102
        return 4 * self.config.n_embd

    @property
    def dropout(self) -> float:  # noqa: D102
        return self.config.dropout

    @property
    def n_layers(self) -> int:  # noqa: D102
        return self.config.n_layer

    @property
    def n_vocab(self) -> int:  # noqa: D102
        return self.config.n_vocab

    @property
    def bias(self) -> bool:  # noqa: D102
        return self.config.bias

    @property
    def num_parameters(self) -> int:  # noqa: D102
        return self.get_num_params(non_embedding=False)

    def __init__(
        self,
        block_size: int,
        bias: bool,
        dropout: float,
        d_model: int,
        n_vocab: int,
        n_layers: int,
        n_heads: int,
    ):
        """Initialize the GPT model.

        Args:
            block_size (int): The block size.
            bias (bool): Whether to include bias parameters.
            dropout (float): The dropout probability.
            d_model (int): The model/embedding dimension.
            n_vocab (int): The number of vocabulary items.
            n_layers (int): The number of layers.
            n_heads (int): The number of attention heads.

        """
        super().__init__()

        # Construct GPTConfig based on Sfirah kwargs
        gpt_config = GPTConfig(
            **{
                "block_size": block_size,
                "vocab_size": n_vocab,
                "n_layer": n_layers,
                "n_head": n_heads,
                "n_embd": d_model,
                "dropout": dropout,
                "bias": bias,
            }
        )

        assert gpt_config.vocab_size is not None
        assert gpt_config.block_size is not None
        self.config = gpt_config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.config.vocab_size, self.config.n_embd),
                wpe=nn.Embedding(self.config.block_size, self.config.n_embd),
                drop=nn.Dropout(self.config.dropout),
                h=nn.ModuleList(
                    [GPTBlock(self.config) for _ in range(self.config.n_layer)]
                ),
                ln_f=GPTLayerNorm(self.config.n_embd, bias=self.config.bias),
            )
        )
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
                )

        # report number of parameters
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model.

        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):  # noqa: D102
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only "
            f"{self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the
            #   very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):  # noqa: D102
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):  # noqa: D102
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in
        #   names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want
        #   to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):  # noqa: D102, E501
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed,
        #   otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and
        #   layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with "
            f"{num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with "
            f"{num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""  # noqa: E501
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()  # noqa: N806
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size  # noqa: N806, E501
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
