from typing import Tuple

import numpy as np

from .tensor import NDArray, Tensor


class Function:
    def forward(self, *args):
        """Computes the forward pass of the operation.
        Args:
            *args: One or more NumPy arrays
        """
        raise NotImplementedError()

    def backward(self, out_grad, node):
        """Calculates backward pass (gradients)
        Args:
            out_grad: upstream gradient flowing from output to input
            node: Value object holding inputs from forward pass
        """
        pass

    def __call__(self, *inputs):  # Takes inputs
        requires_grad = any(t.requires_grad for t in inputs)
        inputs_data = [t.data for t in inputs]  # Gets .data .
        output_data = self.forward(*inputs_data)  # Calls forward.
        # wrap around Tensor
        output_tensor = Tensor(output_data, requires_grad=requires_grad)
        if requires_grad:
            output_tensor._op = self  # Save operation
            output_tensor._inputs = list(inputs)  # Save parents
        return output_tensor


class Add(Function):
    def forward(self, a: NDArray, b: NDArray):
        return a + b

    def backward(self, out_grad, node):
        return out_grad, out_grad


def add(a, b):
    return Add()(a, b)  # `__call__`


class Sub(Function):
    def forward(self, a, b):
        return a - b

    def backward(self, out_grad, node):
        return out_grad, -out_grad

def sub(a, b):
    return Sub()(a, b)

class Mul(Function):
    def forward(self, a, b):
        return a * b

    def backward(self, out_grad, node):
        a, b = node._inputs
        return out_grad * b.data, out_grad * a.data


def mul(a, b):
    return Mul()(a, b)


class Pow(Function):
    def __init__(self, exponent: float):
        self.exponent = exponent

    def forward(self, a: NDArray):
        return a ** self.exponent

    def backward(self, out_grad, node):
        a = node._inputs[0]
        return out_grad * self.exponent * (a.data ** (self.exponent - 1))

def pow(a, exponent: float):
    return Pow(exponent)(a)


class PowerScalar(Function):
    """Raises a tensor to a scalar power: a ** scalar."""
    def __init__(self, scalar: float):
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a ** self.scalar

    def backward(self, out_grad, node):
        a = node._inputs[0]
        return out_grad * self.scalar * (a.data ** (self.scalar - 1))

def power_scalar(a, scalar: float):
    return PowerScalar(scalar)(a)


class ExpBase(Function):
    """Computes base ** a (scalar base raised to tensor power)."""
    def __init__(self, base: float):
        self.base = base

    def forward(self, a: NDArray):
        return self.base ** a

    def backward(self, out_grad, node):
        a = node._inputs[0]
        return out_grad * (self.base ** a.data) * np.log(self.base)

def exp_base(base: float, a):
    return ExpBase(base)(a)
