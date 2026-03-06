import math

from babygrad.tensor import Tensor

GAIN_TABLE = {
    "relu": math.sqrt(2.0),
    "tanh": 5.0 / 3.0,
    "sigmoid": 1.0,
    "gelu": math.sqrt(2.0),
    "silu": math.sqrt(2.0),
    "linear": 1.0,
}


def gain_for_nonlinearity(nonlinearity: str) -> float:
    if nonlinearity not in GAIN_TABLE:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
    return GAIN_TABLE[nonlinearity]


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0,
                   shape=None, **kwargs):
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    if shape is None:
        shape = (fan_in, fan_out)
    return Tensor.rand(*shape, low=-a, high=a, **kwargs)


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0,
                  shape=None, **kwargs):
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    if shape is None:
        shape = (fan_in, fan_out)
    return Tensor.randn(*shape, mean=0, std=std, **kwargs)


def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu",
                    shape=None, **kwargs):
    gain = gain_for_nonlinearity(nonlinearity)
    bound = gain * math.sqrt(3.0 / fan_in)
    if shape is None:
        shape = (fan_in, fan_out)
    return Tensor.rand(*shape, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu",
                   shape=None, **kwargs):
    gain = gain_for_nonlinearity(nonlinearity)
    std = gain / math.sqrt(fan_in)
    if shape is None:
        shape = (fan_in, fan_out)
    return Tensor.randn(*shape, mean=0, std=std, **kwargs)
