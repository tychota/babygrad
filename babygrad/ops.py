from typing import Optional, Tuple

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


class MulScalar(Function):
    """Multiplies a tensor by a scalar: a * scalar."""
    def __init__(self, scalar: float):
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a * self.scalar

    def backward(self, out_grad, node):
        return out_grad * self.scalar

def mul_scalar(a, scalar: float):
    return MulScalar(scalar)(a)


class DivScalar(Function):
    """Divides a tensor by a scalar: a / scalar."""
    def __init__(self, scalar: float):
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a / self.scalar

    def backward(self, out_grad, node):
        return out_grad / self.scalar

def div_scalar(a, scalar: float):
    return DivScalar(scalar)(a)


class TrueDiv(Function):
    def forward(self, a: NDArray, b: NDArray):
        return a / b

    def backward(self, out_grad, node):
        a, b = node._inputs
        grad_a = out_grad / b.data
        grad_b = -out_grad * a.data / (b.data ** 2)
        return grad_a, grad_b

def truediv(a, b):
    return TrueDiv()(a, b)

class Sum(Function):
    def forward(self, a: NDArray, axis: Tuple[int] = None, keepdims: bool = False):
        self.axis = axis
        self.keepdims = keepdims
        return np.sum(a, axis=axis, keepdims=keepdims)

    def backward(self, out_grad, node):
        a = node._inputs[0]
        grad = out_grad
        if not self.keepdims:
            shape = list(a.data.shape)
            if self.axis is None:
                shape = [1] * len(shape)
            else:
                for ax in sorted(self.axis):
                    shape[ax] = 1
            grad = grad.reshape(shape)

        return np.ones_like(a.data) * grad

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a):
        return np.reshape(a, self.shape)

    def backward(self, out_grad, node):
        a = node._inputs[0]
        return reshape(out_grad, a.data.shape)

def reshape(a, shape):
    return Reshape(shape)(a)

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, a):
        if self.axes is None:
            return np.swapaxes(a, -1, -2)

        ndim = a.ndim
        #handling -ve axes
        axes = tuple(ax if ax >= 0 else ndim + ax for ax in self.axes)
        if len(axes) == 2:
            full_axes = list(range(ndim))
            i, j = axes
            full_axes[i], full_axes[j] = full_axes[j], full_axes[i]
            self.full_axes = tuple(full_axes)
        else:
            self.full_axes = axes

        return np.transpose(a, self.full_axes)

    def backward(self, out_grad, node):
        if self.axes is None:
            return transpose(out_grad)
        else:
            inverse_axes = np.argsort(self.axes)
            return transpose(out_grad, tuple(inverse_axes))

def transpose(a, axes=None):
    return Transpose(axes)(a)

class Summation(Function):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def forward(self, a):
        return np.sum(a, axis=self.axes)

    def backward(self, out_grad, node):
        a = node._inputs[0]

        original_shape = a.shape

        if self.axes is None:
            intermediate_shape = (1,) * len(original_shape)
        else:
            axes = self.axes if isinstance(self.axes, (list, tuple)) else (self.axes,)
            axes= [ax if ax >= 0 else ax + len(original_shape) for ax in axes]
            intermediate_shape = list(out_grad.shape)

            #inserting 1's where the axis was vanished.
            for ax in sorted(axes):
                intermediate_shape.insert(ax, 1)
        #reshape
        reshaped_grad = reshape(out_grad, tuple(intermediate_shape))
        ones = np.ones(original_shape)
        ones = Tensor(ones)
        #broadcast or multiply by ones
        return reshaped_grad * ones

def summation(a, axes=None):
    return Summation(axes)(a)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a):
        return np.broadcast_to(a, self.shape)

    def backward(self, out_grad, node):
        a = node._inputs[0]
        original_shape = a.shape
        converted_shape = out_grad.shape

        # Un-prepending
        changed_shape = len(converted_shape) - len(original_shape)
        grad =  out_grad
        for _ in range(changed_shape):
            grad = summation(grad, axes=0)

        # Un-stretching
        for i, (orig_dim, conv_dim) in enumerate(zip(original_shape, grad.shape)):
            if orig_dim == 1 and conv_dim > 1:
                grad = summation(grad, axes=i)
                new_shape = list(grad.shape)
                new_shape.insert(i, 1)
                grad = reshape(grad, tuple(new_shape))

        return grad

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

class MatMul(Function):
    def forward(self, a, b):
        return np.matmul(a, b)

    def backward(self, out_grad, node):
        a, b = node._inputs

        if len(out_grad.shape) == 0:
            out_grad = out_grad.broadcast_to(node.shape)

        grad_a = matmul(out_grad, transpose(b, axes=(-1, -2)))
        grad_b = matmul(transpose(a, axes=(-1, -2)), out_grad)

        while len(grad_a.shape) > len(a.shape):
            grad_a = summation(grad_a, axes=0)
        while len(grad_b.shape) > len(b.shape):
            grad_b = summation(grad_b, axes=0)

        grad_a = grad_a.reshape(a.shape)
        grad_b = grad_b.reshape(b.shape)
        return grad_a, grad_b

def matmul(a, b):
    return MatMul()(a, b)


class ReLU(Function):
    """Applies ReLU element-wise: max(0, x)."""
    def forward(self, a: NDArray):
        return np.maximum(a, 0)

    def backward(self, out_grad, node):
        a = node._inputs[0]
        return out_grad * Tensor((a.data > 0).astype(a.dtype))

def relu(a):
    return ReLU()(a)


class Tanh(Function):
    """Applies tanh element-wise."""
    def forward(self, a: NDArray):
        return np.tanh(a)

    def backward(self, out_grad, node):
        return out_grad * Tensor(1.0 - np.tanh(node._inputs[0].data) ** 2)

def tanh(a):
    return Tanh()(a)


class Sigmoid(Function):
    """Applies sigmoid element-wise: 1 / (1 + exp(-x))."""
    def forward(self, a: NDArray):
        return 1.0 / (1.0 + np.exp(-a))

    def backward(self, out_grad, node):
        s = 1.0 / (1.0 + np.exp(-node._inputs[0].data))
        return out_grad * Tensor(s * (1.0 - s))

def sigmoid(a):
    return Sigmoid()(a)


def gelu(a):
    """GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    return a * Tensor(0.5) * (Tensor(1.0) + tanh(
        Tensor(np.sqrt(2.0 / np.pi)) * (a + Tensor(0.044715) * a ** 3)
    ))


def silu(a):
    """SiLU activation (Swish): x * sigmoid(x)."""
    return a * sigmoid(a)


class Sqrt(Function):
    """Computes element-wise square root: sqrt(x)."""
    def forward(self, a: NDArray):
        return np.sqrt(a)

    def backward(self, out_grad, node):
        a = node._inputs[0]
        return out_grad / (Tensor(2.0 * np.sqrt(a.data)))

def sqrt(a):
    return Sqrt()(a)


class LogSumExp(Function):
    """Computes log(sum(exp(x), axis)) with the max trick for stability."""
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, a: NDArray):
        self.max_val = np.max(a, axis=self.axes, keepdims=True)
        shifted = a - self.max_val
        self.exp_shifted = np.exp(shifted)
        sum_exp = np.sum(self.exp_shifted, axis=self.axes, keepdims=True)
        result = np.log(sum_exp) + self.max_val
        # Remove the kept dims to match expected output shape
        if self.axes is not None:
            result = result.squeeze(axis=self.axes)
        else:
            result = result.squeeze()
        return result

    def backward(self, out_grad, node):
        a = node._inputs[0]
        # softmax = exp(a - max) / sum(exp(a - max))
        max_val = np.max(a.data, axis=self.axes, keepdims=True)
        shifted = a.data - max_val
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=self.axes, keepdims=True)
        softmax = exp_shifted / sum_exp

        # Expand out_grad to match input shape
        if self.axes is not None:
            grad = out_grad.data
            for ax in sorted(self.axes if isinstance(self.axes, (list, tuple)) else (self.axes,)):
                grad = np.expand_dims(grad, axis=ax)
        else:
            grad = out_grad.data.reshape([1] * len(a.data.shape))

        return Tensor(grad * softmax)

def logsumexp(a, axes=None):
    return LogSumExp(axes)(a)


class Exp(Function):
    """Computes e^x element-wise."""
    def forward(self, a: NDArray):
        self.out = np.exp(a)
        return self.out
    def backward(self, out_grad, node):
        return out_grad * Tensor(self.out)

def exp(a):
    return Exp()(a)


class Log(Function):
    """Computes ln(x) element-wise."""
    def forward(self, a: NDArray):
        return np.log(a)
    def backward(self, out_grad, node):
        a = node._inputs[0]
        return out_grad / a

def log(a):
    return Log()(a)


class Max(Function):
    """Computes max along an axis."""
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, a: NDArray):
        return np.max(a, axis=self.axis, keepdims=self.keepdims)
    def backward(self, out_grad, node):
        a = node._inputs[0]
        if not self.keepdims and self.axis is not None:
            expanded = np.expand_dims(out_grad.data, axis=self.axis)
        elif not self.keepdims and self.axis is None:
            expanded = out_grad.data.reshape([1] * a.data.ndim)
        else:
            expanded = out_grad.data
        max_val = np.max(a.data, axis=self.axis, keepdims=True)
        mask = (a.data == max_val).astype(np.float32)
        mask = mask / np.sum(mask, axis=self.axis, keepdims=True)
        return Tensor(expanded * mask)

def max(a, axis=None, keepdims=False):
    return Max(axis, keepdims)(a)


def softmax(a, axis=-1):
    """Numerically stable softmax: exp(x - max) / sum(exp(x - max))."""
    m = max(a, axis=axis, keepdims=True)
    shifted = a - m
    e = exp(shifted)
    s = summation(e, axes=(axis,))
    new_shape = list(a.shape)
    new_shape[axis] = 1
    s_reshaped = reshape(s, tuple(new_shape))
    return e / broadcast_to(s_reshaped, a.shape)


class Where(Function):
    """Element-wise conditional: where(cond, a, b)."""
    def forward(self, cond: NDArray, a: NDArray, b: NDArray):
        self.cond = cond
        return np.where(cond, a, b)
    
    def backward(self, out_grad, node):
        cond = self.cond
        grad_a = out_grad * Tensor(cond.astype(np.float32))
        grad_b = out_grad * Tensor((1 - cond).astype(np.float32))
        return None, grad_a, grad_b

def where(cond, a, b):
    return Where()(cond, a, b)


class Tril(Function):
    """Lower triangular mask."""
    def __init__(self, k=0):
        self.k = k
    def forward(self, a: NDArray):
        return np.tril(a, k=self.k)
    def backward(self, out_grad, node):
        return Tensor(np.tril(out_grad.data, k=self.k))

def tril(a, k=0):
    return Tril(k)(a)


class Concat(Function):
    """Concatenate tensors along an axis."""
    def __init__(self, axis=0):
        self.axis = axis
    def forward(self, *arrays):
        self.sizes = [a.shape[self.axis] for a in arrays]
        return np.concatenate(arrays, axis=self.axis)
    def backward(self, out_grad, node):
        splits = np.cumsum(self.sizes[:-1])
        grads = np.split(out_grad.data, splits, axis=self.axis)
        return tuple(Tensor(g) for g in grads)

def concat(tensors, axis=0):
    return Concat(axis)(*tensors)


class SliceOp(Function):
    """Internal op for split backward."""
    def __init__(self, axis, start, stop, original_shape):
        self.axis = axis
        self.start = start
        self.stop = stop
        self.original_shape = original_shape
    def forward(self, a: NDArray):
        raise NotImplementedError
    def backward(self, out_grad, node):
        grad = np.zeros(self.original_shape, dtype=np.float32)
        idx = [slice(None)] * len(self.original_shape)
        idx[self.axis] = slice(self.start, self.stop)
        grad[tuple(idx)] = out_grad.data
        return Tensor(grad)

def split(a, sections, axis=0):
    size = a.shape[axis]
    chunk_size = size // sections
    parts = []
    for i in range(sections):
        idx = [slice(None)] * a.data.ndim
        idx[axis] = slice(i * chunk_size, (i + 1) * chunk_size)
        sliced_data = a.data[tuple(idx)]
        t = Tensor(sliced_data, requires_grad=a.requires_grad)
        if a.requires_grad:
            t._op = SliceOp(axis, i * chunk_size, (i + 1) * chunk_size, a.shape)
            t._inputs = [a]
        parts.append(t)
    return parts


class Gather(Function):
    """Gather elements along an axis by index."""
    def __init__(self, axis=0):
        self.axis = axis
    def forward(self, a: NDArray, indices: NDArray):
        self.indices = indices.astype(int)
        return np.take(a, self.indices, axis=self.axis)
    def backward(self, out_grad, node):
        a = node._inputs[0]
        grad = np.zeros_like(a.data)
        if self.axis == 0:
            np.add.at(grad, self.indices, out_grad.data)
        else:
            idx = tuple(slice(None) if i != self.axis else self.indices
                        for i in range(a.data.ndim))
            np.add.at(grad, idx, out_grad.data)
        return Tensor(grad), None

def gather(a, indices, axis=0):
    indices_tensor = Tensor(indices) if not isinstance(indices, Tensor) else indices
    return Gather(axis)(a, indices_tensor)
