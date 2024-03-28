"""Flax implementation of Transformer model."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct
from jax import lax


@struct.dataclass
class TransformerConfig:
    """Config for Transformer model."""

    vocab_size: int
    output_vocab_size: int
    share_embeddings: bool = False
    logits_via_embedding: bool = False
    dtype: Any = jnp.float32
    emb_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 2048
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    deterministic: bool = False
    decode: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    posemb_init: Callable | None = None


def shift_right(x, axis=1):  # noqa: D103
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))

    return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis=axis)


def shift_inputs(x, segment_ids=None, axis=1):  # noqa: D103
    shifted = shift_right(x, axis=axis)
    if segment_ids is not None:
        shifted *= segment_ids == shift_right(segment_ids, axis=axis)
    return shifted


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10_000.0):  # noqa: D103
    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, : d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class AddPositionEmbs(nn.Module):
    """Positional embeddings."""

    config: TransformerConfig
    decode: bool = False

    @nn.compact
    def __call__(self, inputs, inputs_positions=None):  # noqa: D102
        config = self.config
        assert (
            inputs.ndim == 3
        ), f"inputs shape must be (batch, len, d_model), got {inputs.shape}"

        length = inputs.shape[1]
        pos_emb_shape = (1, config.max_len, inputs.shape[-1])
        if config.posemb_init is None:
            pos_embedding = sinusoidal_init(max_len=config.max_len)(
                None, pos_emb_shape, None
            )
        else:
            pos_embedding = self.param(
                "pos_embedding", config.posemb_init, pos_emb_shape
            )
        pe = pos_embedding[:, :length, :]

        if self.decode:
            is_initialized = self.has_variable("cache", "cache_index")
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.uint32)
            )
            if is_initialized:
                i = cache_index.value
                cache_index.value = i + 1
                _, _, df = pos_embedding.shape
                pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))
        if inputs_positions is None:
            # normal unpacked case:
            return inputs + pe
        else:
            # for packed data we need to use known position indices:
            return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MLPBlock(nn.Module):
    """MLP block."""

    config: TransformerConfig
    out_dim: int | None = None

    @nn.compact
    def __call__(self, inputs):  # noqa: D102
        config = self.config
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            config.mlp_dim,
            dtype=config.dtype,
            kernel_init=nn.with_logical_partitioning(
                config.kernel_init, ("embed", "mlp")
            ),
            bias_init=nn.with_logical_partitioning(config.bias_init, ("mlp",)),
        )(inputs)
        x = nn.relu(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        output = nn.Dense(
            actual_out_dim,
            dtype=config.dtype,
            kernel_init=nn.with_logical_partitioning(
                config.kernel_init, ("mlp", "embed")
            ),
            bias_init=nn.with_logical_partitioning(config.bias_init, ("embed",)),
        )(x)
        output = nn.Dropout(rate=config.dropout_rate)(
            output, deterministic=config.deterministic
        )
        return output


class TransformerBlock(nn.Module):
    """Transformer block."""

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, decoder_mask=None) -> Any:  # noqa: D102
        config = self.config

        assert inputs.ndim == 3
        x = nn.LayerNorm(
            dtype=config.dtype,
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
            scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
        )(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=nn.with_logical_partitioning(
                config.kernel_init, ("embed", "kv")
            ),
            bias_init=nn.with_logical_partitioning(config.bias_init, ("embed",)),
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=config.deterministic,
            decode=config.decode,
        )(x, mask=decoder_mask)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        x = x + inputs

        z = nn.LayerNorm(
            dtype=config.dtype,
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
            scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
        )(x)
        z = MLPBlock(config=config)(z)

        return x + z


class Decoder(nn.Module):
    """Decoder."""

    config: TransformerConfig
    shared_embedding: Any = None

    @nn.compact
    def __call__(  # noqa: D102
        self,
        inputs,
        inputs_positions=None,
        inputs_segmentation=None,
        decoder_mask=None,
    ):
        config = self.config
        assert inputs.ndim == 2  # (batch, len)

        if self.shared_embedding is None:
            output_embed = nn.Embed(
                num_embeddings=config.output_vocab_size,
                features=config.emb_dim,
                embedding_init=nn.with_logical_partitioning(
                    nn.initializers.normal(stddev=1.0),
                    (
                        "vocab",
                        "embed",
                    ),
                ),
            )
        else:
            output_embed = self.shared_embedding

        y = inputs.astype("int32")
        if not config.decode:
            y = shift_inputs(y, segment_ids=inputs_segmentation)
        y = output_embed(y)
        y = AddPositionEmbs(
            config=config, decode=config.decode, name="posembed_output"
        )(y, inputs_positions=inputs_positions)
        y = nn.Dropout(rate=config.dropout_rate)(y, deterministic=config.deterministic)

        y = y.astype(config.dtype)

        for lyr in range(config.num_layers):
            y = TransformerBlock(config=config, name=f"encoderdecoderblock_{lyr}")(
                y, decoder_mask=decoder_mask
            )
        y = nn.LayerNorm(
            dtype=config.dtype,
            name="encoderdecoder_norm",
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed",)),
            scale_init=nn.with_logical_partitioning(nn.initializers.ones, ("embed",)),
        )(y)

        if config.logits_via_embedding:
            # Use the transpose of embedding matrix for logit transform.
            logits = output_embed.attend(y.astype(jnp.float32))
            # Correctly normalize pre-softmax logits for this shared case.
            logits = logits / jnp.sqrt(y.shape[-1])
        else:
            logits = nn.Dense(
                config.output_vocab_size,
                dtype=config.dtype,
                kernel_init=nn.with_logical_partitioning(
                    config.kernel_init, ("embed", "vocab")
                ),
                bias_init=nn.with_logical_partitioning(config.bias_init, ("vocab",)),
                name="logitdense",
            )(y)
        return logits


class CausalLM(nn.Module):
    """Causal language model."""

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, inputs_positions=None, inputs_segmentation=None):  # noqa: D102, E501
        config = self.config
        assert inputs.ndim == 2

        if config.decode:
            decoder_mask = None
        else:
            decoder_mask = nn.combine_masks(
                nn.make_attention_mask(inputs > 0, inputs > 0, dtype=config.dtype),
                nn.make_causal_mask(inputs, dtype=config.dtype),
            )

        if inputs_segmentation is not None:
            decoder_mask = nn.combine_masks(
                decoder_mask,
                nn.make_attention_mask(
                    inputs_segmentation,
                    inputs_segmentation,
                    jnp.equal,
                    dtype=config.dtype,
                ),
            )

        logits = Decoder(config=config, shared_embedding=None, name="decoder")(
            inputs,
            inputs_positions=inputs_positions,
            inputs_segmentation=inputs_segmentation,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=None,
        )
        return logits.astype(self.config.dtype)
