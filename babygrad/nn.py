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
