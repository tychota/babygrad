from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .ops import Function

NDArray = np.ndarray

def _ensure_tensor(val):
    return val if isinstance(val, Tensor) else Tensor(val,
     requires_grad=False)


class Tensor:
    def __init__(self, data, *, device=None, dtype="float32", requires_grad=True):
        """
        Create a new tensor.
        Args:
            data: Array-like data (list, numpy array, or another Tensor)
            device: Device placement (currently ignored, CPU only)
            dtype: Data type for the array
            requires_grad: Whether to track gradients for this tensor
        """
        if isinstance(data, Tensor):
            if dtype is None:
                dtype = data.dtype
            self.data = data.numpy().astype(dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype if dtype is not None else data.dtype)
        else:
            self.data = np.array(data, dtype=dtype if dtype is not None else "float32")

        self.grad: Optional[NDArray] = None
        self.requires_grad = requires_grad
        self._op: Optional["Function"] = None
        self._inputs: List["Tensor"] = []
        self._device = device if device else "cpu"

    def __repr__(self):
        """
        Detailed representation showing data and gradient tracking.
        Example:
            >>> x = Tensor([1, 2, 3])
            >>> print(repr(x))
            Tensor([1. 2. 3.], requires_grad=True)
        """
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self):
        """
        Simple string representation (just the data).
        Example:
            >>> x = Tensor([1, 2, 3])
            >>> print(x)
            [1. 2. 3.]
        """
        return str(self.data)

    def backward(self, out_grad=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")

        # Build a family tree of tensors in topological order
        topo_order = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._inputs:
                    build_topo(parent)
                topo_order.append(tensor)
        build_topo(self)

        # Initialize the ledger
        grads = {}
        if out_grad is None:
            # The "output" gradient: dL/dL = 1
            grads[id(self)] = Tensor(np.ones_like(self.data))
        else:
            grads[id(self)] = _ensure_tensor(out_grad)

        # Walk the Graph Backwards
        for node in reversed(topo_order):
            out_grad = grads.get(id(node))
            if out_grad is None:
                continue

            # Store the final result in the .grad attribute
            if node.grad is None:
                node.grad = np.array(out_grad.data, copy=True)
            else:
                node.grad += out_grad.data

            # Propagate to Parents
            if node._op:
                input_grads = node._op.backward(out_grad, node)
                if not isinstance(input_grads, tuple):
                    input_grads = (input_grads,)

                for i, parent in enumerate(node._inputs):
                    if parent.requires_grad:
                        parent_id = id(parent)
                        if parent_id not in grads:
                            # First time seeing this parent
                            grads[parent_id] = input_grads[i]
                        else:
                            #  Sum the gradients!
                            grads[parent_id] = grads[parent_id] + input_grads[i]

    def numpy(self):
        """
        Return the data as a NumPy array (detached from the autograd graph).
        This returns a copy, so modifying the result will not affect
        the tensor's data.
        Examples:
            >>> x = Tensor([1, 2, 3])
            >>> y = x + 1   # y is still a Tensor, part of the graph
            >>> z = x.numpy() + 1  # z is a NumPy array, not part of the graph
        Returns:
            np.ndarray: A copy of the tensor's data as a NumPy array.
        """
        return self.data.copy()

    def detach(self):
        """
        Creates a new Tensor with same data but no gradient tracking.
        Useful when you want to use values without building
        computation graph.
        Returns:
            Tensor: New tensor with requires_grad=False
        Example:
            >>> x = Tensor([1, 2, 3], requires_grad=True)
            >>> y = x.detach()  # y doesn't track gradients
            >>> z = y * 2       # This operation won't be in graph
        """
        return Tensor(self.data, requires_grad=False)

    @property
    def shape(self):
        """Shape of the tensor."""
        return self.data.shape

    @property
    def dtype(self):
        """Data type of the tensor."""
        return self.data.dtype

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.data.ndim

    @property
    def size(self):
        """Total number of elements."""
        return self.data.size

    @property
    def op(self):
        """What Math operation created this Tensor."""
        return self._op

    @property
    def inputs(self):
        """What inputs created this Tensor."""
        return self._inputs

    @property
    def device(self):
        """Device where tensor lives."""
        return self._device

    def __add__(self, other):
        """Addition: a + b"""
        from .ops import Add
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add()(self, other)

    def __radd__(self, other):
        """Right addition: 5 + tensor"""
        return self.__add__(other)

    def __mul__(self, other):
        """Multiplication: a * b"""
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float)):
                from .ops import MulScalar
                return MulScalar(float(other))(self)
            other = Tensor(other)
        from .ops import Mul
        return Mul()(self, other)

    def __rmul__(self, other):
        """Right multiplication: 5 * tensor"""
        return self.__mul__(other)

    def __neg__(self):
        """Negation: -a"""
        return self * (-1)

    def __sub__(self, other):
        """Subtraction: a - b"""
        from .ops import Sub
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Sub()(self, other)

    def __rsub__(self, other):
        """Right subtraction: 5 - tensor"""
        from .ops import Sub
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Sub()(other, self)

    def __pow__(self, exponent: float):
        """Power: a ** exponent"""
        from .ops import PowerScalar
        return PowerScalar(float(exponent))(self)

    def __rpow__(self, base: float):
        """Right power: base ** a"""
        from .ops import ExpBase
        return ExpBase(base)(self)

    def __truediv__(self, other):
        """Division: a / b"""
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float)):
                from .ops import DivScalar
                return DivScalar(float(other))(self)
            other = Tensor(other)
        from .ops import TrueDiv
        return TrueDiv()(self, other)

    def __rtruediv__(self, other):
        """Right division: 5 / tensor"""
        from .ops import TrueDiv
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return TrueDiv()(other, self)

    def __matmul__(self, other):
        """Matrix multiplication: a @ b"""
        from .ops import MatMul
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return MatMul()(self, other)

    def matmul(self, other):
        """Explicit method for matrix multiplication: a.matmul(b)"""
        return self.__matmul__(other)

    def reshape(self, *shape, **kwargs):
        """Return a reshaped view of this tensor."""
        from .ops import reshape
        return reshape(self, shape)

    def broadcast_to(self, shape):
        """Broadcast this tensor to the given shape."""
        from .ops import broadcast_to
        return broadcast_to(self, shape)

    @staticmethod
    def randn(*shape, dtype="float32"):
        """Create a tensor with random normal values."""
        return Tensor(np.random.randn(*shape).astype(dtype))

    @staticmethod
    def zeros(*shape, dtype="float32"):
        """Create a tensor of zeros."""
        return Tensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def ones(*shape, dtype="float32"):
        """Create a tensor of ones."""
        return Tensor(np.ones(shape, dtype=dtype))

    @staticmethod
    def randb(*shape, p=0.5, dtype="float32"):
        """Create a tensor of random binary values (0 or 1) with P(1) = p."""
        return Tensor(np.random.binomial(1, p, size=shape).astype(dtype))
