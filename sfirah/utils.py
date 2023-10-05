import logging

from torch import nn
from torch.nn import functional as F  # noqa: N812

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _get_funcational_activation(activation: str) -> callable:
    activation_funcs = {
        "celu": F.celu,
        "elu": F.elu,
        "gelu": F.gelu,
        "glu": F.glu,
        "hardshrink": F.hardshrink,
        "hardsigmoid": F.hardsigmoid,
        "hardswish": F.hardsigmoid,
        "hardtanh": F.hardtanh,
        "leaky_relu": F.leaky_relu,
        "logsigmoid": F.logsigmoid,
        "log_softmax": F.log_softmax,
        "mish": F.mish,
        "prelu": F.prelu,
        "relu": F.relu,
        "relu6": F.relu6,
        "rrelu": F.rrelu,
        "selu": F.selu,
        "sigmoid": F.sigmoid,
        "silu": F.silu,
        "softmax": F.softmax,
        "softmin": F.softmin,
        "softplus": F.softplus,
        "softshrink": F.softshrink,
        "softsign": F.softsign,
        "tanh": F.tanh,
        "tanhshrink": F.tanhshrink,
    }

    if activation not in activation_funcs:
        raise ValueError(
            f"Unknown activation `{activation}`. Must be one of: "
            f"{list(activation_funcs.keys())}"
        )

    if activation not in ["relu", "gelu"]:
        logger.warning(
            f"PyTorch does not optimize for using `{activation}`; "
            "Consider using `relu` or `gelu` instead."
        )

    return activation_funcs[activation]


def _get_module_activation(activation: str) -> nn.Module:
    activation_mods = {
        "celu": nn.CELU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "glu": nn.GLU,
        "hardshrink": nn.Hardshrink,
        "hardsigmoid": nn.Hardsigmoid,
        "hardswish": nn.Hardswish,
        "hardtanh": nn.Hardtanh,
        "leaky_relu": nn.LeakyReLU,
        "logsigmoid": nn.LogSigmoid,
        "log_softmax": nn.LogSoftmax,
        "mish": nn.Mish,
        "prelu": nn.PReLU,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "rrelu": nn.RReLU,
        "selu": nn.SELU,
        "sigmoid": nn.Sigmoid,
        "silu": nn.SiLU,
        "softmax": nn.Softmax,
        "softmin": nn.Softmin,
        "softplus": nn.Softplus,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanh": nn.Tanh,
        "tanhshrink": nn.Tanhshrink,
    }

    if activation not in activation_mods:
        raise ValueError(
            f"Unknown activation `{activation}`. Must be one of: "
            f"{list(activation_mods.keys())}"
        )

    if activation not in ["relu", "gelu"]:
        logger.warning(
            f"PyTorch does not optimize for using `{activation}`; "
            "Consider using `relu` or `gelu` instead."
        )

    return activation_mods[activation]()


def get_activation(activation: str, functional: bool = False) -> callable | nn.Module:
    if functional:
        return _get_funcational_activation(activation)
    else:
        return _get_module_activation(activation)
