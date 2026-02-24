from typing import Any, List, Optional

import numpy as np

from babygrad.tensor import Tensor
from babygrad import ops


class Parameter(Tensor):
    """
    A special Tensor that tells a Module it is a learnable parameter.

    Example:
        >>> self.weights = Parameter(Tensor.randn(10, 5))
    """

    def __init__(self, data, *args, **kwargs):
        kwargs["requires_grad"] = True
        super().__init__(data, *args, **kwargs)


def _get_parameters(data) -> List[Parameter]:
    """Recursively find all Parameter instances in data."""
    params = []
    if isinstance(data, Parameter):
        return [data]
    if isinstance(data, Module):
        return data.parameters()
    if isinstance(data, dict):
        for value in data.values():
            params.extend(_get_parameters(value))
    if isinstance(data, (list, tuple)):
        for item in data:
            params.extend(_get_parameters(item))
    return params


def _get_modules(obj) -> list["Module"]:
    """Recursively find all Module instances in data."""
    modules = []
    if isinstance(obj, Module):
        return [obj]
    if isinstance(obj, dict):
        for value in obj.values():
            modules.extend(_get_modules(value))
    if isinstance(obj, (list, tuple)):
        for item in obj:
            modules.extend(_get_modules(item))
    return modules


class Module:
    """
    Base class for all neural network layers.

    Example:
        class MyLayer(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(Tensor([1.0, 2.0]))

            def forward(self, x):
                return x * self.weight
    """

    def __init__(self):
        self.training = True

    def parameters(self) -> List[Parameter]:
        """Returns a list of all unique parameters in the module and its submodules."""
        params = _get_parameters(self.__dict__)
        unique_params = []
        seen_ids = set()
        for p in params:
            if id(p) not in seen_ids:
                unique_params.append(p)
                seen_ids.add(id(p))
        return unique_params

    def forward(self, *args, **kwargs):
        """The forward pass logic. Must be defined by subclasses."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Makes the module callable, delegating to forward()."""
        return self.forward(*args, **kwargs)

    def train(self):
        """Set this module and all child modules to training mode."""
        self.training = True
        for m in _get_modules(self.__dict__):
            m.training = True

    def eval(self):
        """Set this module and all child modules to evaluation mode."""
        self.training = False
        for m in _get_modules(self.__dict__):
            m.training = False


class ReLU(Module):
    """Applies ReLU element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    """Applies Tanh element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    """Applies Sigmoid element-wise."""

    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)


class Flatten(Module):
    """Flattens a tensor to (batch_size, -1)."""

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        flat_dim = int(np.prod(x.shape[1:]))
        return x.reshape(batch_size, flat_dim)


class Linear(Module):
    """Applies a linear transformation: y = xW + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any | None = None,
        dtype: str = "float32",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor.randn(in_features, out_features))
        self.bias = None
        if bias:
            self.bias = Parameter(Tensor.zeros(1, out_features))

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out += self.bias.broadcast_to(out.shape)
        return out


class Sequential(Module):
    """Chains a sequence of modules: output of one becomes input of next."""

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class Residual(Module):
    """Residual connection: f(x) + x."""

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


class Dropout(Module):
    """Randomly zeroes elements during training, scaled by 1/(1-p)."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = Tensor.randb(*x.shape, p=(1 - self.p))
            return (x * mask) / (1 - self.p)
        else:
            return x


class LayerNorm1d(Module):
    """Applies Layer Normalization over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(Tensor.ones(dim, dtype=dtype))
        self.bias = Parameter(Tensor.zeros(dim, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, dim)
        sum_x = ops.summation(x, axes=(1,))
        mean = sum_x / self.dim
        mean_reshaped = ops.reshape(mean, (x.shape[0], 1))
        mean_broadcasted = ops.broadcast_to(mean_reshaped, x.shape)
        x_minus_mean = x - mean_broadcasted
        var = ops.summation(x_minus_mean**2, axes=(1,)) / self.dim
        var_reshaped = ops.reshape(var, (x.shape[0], 1))
        var_broadcasted = ops.broadcast_to(var_reshaped, x.shape)
        std = ops.sqrt(var_broadcasted + self.eps)
        x_hat = x_minus_mean / std
        weight_reshaped = ops.reshape(self.weight, (1, self.dim))
        bias_reshaped = ops.reshape(self.bias, (1, self.dim))
        weight_broadcasted = ops.broadcast_to(weight_reshaped, x.shape)
        bias_broadcasted = ops.broadcast_to(bias_reshaped, x.shape)
        return weight_broadcasted * x_hat + bias_broadcasted


class BatchNorm1d(Module):
    """Applies Batch Normalization over the batch dimension."""

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1,
                 device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(Tensor.ones(dim, dtype=dtype))
        self.bias = Parameter(Tensor.zeros(dim, dtype=dtype))
        self.running_mean = Tensor.zeros(dim, dtype=dtype)
        self.running_var = Tensor.ones(dim, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # x.shape is (batch_size, dim)
            batch_size = x.shape[0]
            # (batch_size,dim) -> (dim,)
            mean = ops.summation(x, axes=(0,)) / batch_size

            # (bs,dim) - (bs,dim) -> (dim,)
            var = ops.summation((x - ops.broadcast_to(ops.reshape(mean,
                        (1, self.dim)), x.shape)) ** 2, axes=(0,)) / batch_size

            self.running_mean.data = (1 - self.momentum) * \
                            self.running_mean.data + self.momentum * mean.data

            self.running_var.data = (1 - self.momentum) * \
                            self.running_var.data + self.momentum * var.data

            mean_to_use = mean
            var_to_use = var
        else:
            mean_to_use = self.running_mean
            var_to_use = self.running_var

        # mean_to_use (dim,) -> (1,dim)
        mean_reshaped = ops.reshape(mean_to_use, (1, self.dim))

        # var_to_use (dim,) -> (1,dim)
        var_reshaped = ops.reshape(var_to_use, (1, self.dim))

        std = ops.sqrt(var_reshaped + self.eps)

        # (bs,dim) - (bs,dim) / (bs,dim)
        x_hat = (x - ops.broadcast_to(mean_reshaped, x.shape)) \
                     / ops.broadcast_to(std, x.shape)

        # weight/bias (dim,) -> (1,dim)
        weight_reshaped = ops.reshape(self.weight, (1, self.dim))
        bias_reshaped = ops.reshape(self.bias, (1, self.dim))

        # (1,dim) -> (bs,dim)
        return ops.broadcast_to(weight_reshaped, x.shape) * x_hat \
                     + ops.broadcast_to(bias_reshaped, x.shape)


class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Calculates the Mean Squared Error."""
        diff = pred - target
        sq_diff = diff * diff
        return sq_diff.sum() / Tensor(target.data.size)
