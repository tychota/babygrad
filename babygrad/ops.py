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


class Mul(Function):
    def forward(self, a, b):
        return a * b

    def backward(self, out_grad, node):
        a, b = node._inputs
        return out_grad * b.data, out_grad * a.data


def mul(a, b):
    return Mul()(a, b)
