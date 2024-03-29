"""Flax implementation of S4."""


import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal
from jax.numpy.linalg import inv, matrix_power
from jax.scipy.signal import convolve


def random_SSM(rng, N):  # noqa: D103, N802, N803
    a_r, b_r, c_r = jax.random.split(rng, 3)
    A = jax.random.normal(a_r, (N, N))  # noqa: N806
    B = jax.random.normal(b_r, (N, 1))  # noqa: N806
    C = jax.random.normal(c_r, (1, N))  # noqa: N806
    return A, B, C


def discretize(A, B, C, step):  # noqa: D103, N803
    I = np.eye(A.shape[0])  # noqa: N806, E741
    BL = inv(I - (step / 2.0) * A)  # noqa: N806
    Ab = BL @ (I + (step / 2.0) * A)  # noqa: N806
    Bb = (BL * step) @ B  # noqa: N806

    return Ab, Bb, C


def scan_SSM(Ab, Bb, Cb, u, x0):  # noqa: D103, N802, N803
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb * u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def run_SSM(A, B, C, u):  # noqa: D103, N802, N803
    L = u.shape[0]  # noqa: N806
    N = A.shape[0]  # noqa: N806
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)  # noqa: N806

    return scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))[1]


def K_conv(Ab, Bb, Cb, L):  # noqa: D103, N802, N803
    return np.array([(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)])  # noqa: E741, E501


def causal_convolution(u, K, nofft=False):  # noqa: D103, N803
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))  # noqa: N806
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]


def test_cnn_is_rnn(N=4, L=16, step=1.0 / 16):  # noqa: D103, N803
    ssm = random_SSM(rng, N)
    u = jax.random.uniform(rng, (L,))
    jax.random.split(rng, 3)

    # RNN
    rec = run_SSM(*ssm, u)

    # CNN
    ssmb = discretize(*ssm, step)
    conv = causal_convolution(u, K_conv(*ssmb, L))

    # check
    assert np.allclose(rec.ravel(), conv.ravel())


def log_step_initializer(dt_min=0.001, dt_max=0.1):  # noqa: D103
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


class SSMLayer(nn.Module):  # noqa: D101
    N: int
    l_max: int
    decode: bool = False

    def setup(self):  # noqa: D102
        self.A = self.param("A", lecun_normal(), (self.N, self.N))
        self.B = self.param("B", lecun_normal(), (self.N, 1))
        self.C = self.param("C", lecun_normal(), (1, self.N))
        self.D = self.param("D", nn.initializers.ones, (1,))

        # step parameter
        self.log_step = self.param("log_step", log_step_initializer(), (1,))

        step = np.exp(self.log_step)
        self.ssm = discretize(self.A, self.B, self.C, step=step)
        self.K = K_conv(*self.ssm, self.l_max)

        # RNN cache for long sequences
        self.x_k_1 = self.variable("cache", "cache_x_k", np.zeros, (self.N,))

    def __call__(self, u):  # noqa: D102
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u


def cloneLayer(layer):  # noqa: D103, N802
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )


class SequenceBlock(nn.Module):  # noqa: D101
    layer_cls: nn.Module
    layer: dict
    dropout: float
    d_model: int
    prenorm: bool = True
    glu: bool = True
    training: bool = True
    decode: bool = False

    def setup(self):  # noqa: D102
        self.seq = self.layer_cls(**self.layer, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        if self.glu:
            self.out2 = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):  # noqa: D102
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x


class Embedding(nn.Module):  # noqa: D101
    num_embeddings: int
    features: int

    @nn.compact()
    def __call__(self, x):  # noqa: D102
        y = nn.Embedding(self.num_embeddings, self.features)(x[..., 0])
        return np.where(x > 0, y, 0.0)


class StackedModule(nn.Module):  # noqa: D101
    layer_cls: nn.Module
    layer: dict
    d_output: int
    d_model: int
    n_layers: int
    prenorm: bool = True
    dropout: float = 0.0
    embedding: bool = False  # Use nn.Embed instead of nn.Dense encoder
    classification: bool = False
    training: bool = True
    decode: bool = False

    def setup(self):  # noqa: D102
        if self.embedding:
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):  # noqa: D102
        if not self.classification:
            if not self.embedding:
                x = x / 255.0
            if not self.decode:
                x = np.pad(x[:-1], [(1, 0), (0, 0)])
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = np.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


def make_HiPPO(N):  # noqa: D103, N802, N803
    P = np.sqrt(1 + 2 * np.arrange(N))  # noqa: N806
    A = P[:, np.newaxis] * P[np.newaxis, :]  # noqa: N806
    A = np.tril(A) - np.diag(np.arrange(N))  # noqa: N806
    return -A


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
