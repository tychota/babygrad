import numpy as np

NDArray = np.ndarray

def _ensure_tensor(val):
    return val if isinstance(val, Tensor) else Tensor(val,
     requires_grad=False)


class Tensor:
    def __init__(self, data, *, device=None, dtype="float32",
     requires_grad=False):
        if isinstance(data, Tensor):
            if dtype is None:
                dtype = data.dtype
            self.data = data.numpy().astype(dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype if dtype is not None else data.dtype)
        else:
            self.data = np.array(data, dtype=dtype if dtype is not None
             else "float32")
        self.grad = None
        self.requires_grad = requires_grad
        self._op = None
        self._inputs = []
        self._device = device if device else "cpu"

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self):
        return str(self.data)

    def backward(self, out_grad=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")

        topo_order = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._inputs:
                    build_topo(parent)
                topo_order.append(tensor)
        build_topo(self)

        grads = {}
        if out_grad is None:
            grads[id(self)] = Tensor(np.ones_like(self.data))
        else:
            grads[id(self)] = _ensure_tensor(out_grad)

        for node in reversed(topo_order):
            out_grad = grads.get(id(node))
            if out_grad is None:
                continue

            if node.grad is None:
                node.grad = np.array(out_grad.data, copy=True)
            else:
                node.grad += out_grad.data

            if node._op:
                input_grads = node._op.backward(out_grad, node)
                if not isinstance(input_grads, tuple):
                    input_grads = (input_grads,)

                for i, parent in enumerate(node._inputs):
                    if parent.requires_grad:
                        parent_id = id(parent)
                        if parent_id not in grads:
                            grads[parent_id] = input_grads[i]
                        else:
                            grads[parent_id] = grads[parent_id] + input_grads[i]

    def numpy(self):
        return self.data.copy()

    def detach(self):
        return Tensor(self.data, requires_grad=False, dtype=str(self.dtype))

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def op(self):
        return self._op

    @property
    def inputs(self):
        return self._inputs

    @property
    def device(self):
        return self._device

    @property
    def T(self):
        return self.transpose()

    def __add__(self, other):
        from .ops import Add
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add()(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float)):
                from .ops import MulScalar
                return MulScalar(float(other))(self)
            other = Tensor(other)
        from .ops import Mul
        return Mul()(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        from .ops import Sub
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Sub()(self, other)

    def __rsub__(self, other):
        from .ops import Sub
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Sub()(other, self)

    def __pow__(self, exponent: float):
        from .ops import PowerScalar
        return PowerScalar(float(exponent))(self)

    def __rpow__(self, base: float):
        from .ops import ExpBase
        return ExpBase(base)(self)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            if isinstance(other, (int, float)):
                from .ops import DivScalar
                return DivScalar(float(other))(self)
            other = Tensor(other)
        from .ops import TrueDiv
        return TrueDiv()(self, other)

    def __rtruediv__(self, other):
        from .ops import TrueDiv
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return TrueDiv()(other, self)

    def __matmul__(self, other):
        from .ops import MatMul
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return MatMul()(self, other)

    def matmul(self, other):
        return self.__matmul__(other)

    def reshape(self, *shape, **kwargs):
        from .ops import reshape
        return reshape(self, shape)

    def broadcast_to(self, shape):
        from .ops import broadcast_to
        return broadcast_to(self, shape)

    def transpose(self, axes=None):
        from .ops import transpose
        return transpose(self, axes)

    def sum(self, axes=None):
        from .ops import summation
        return summation(self, axes=axes)

    @classmethod
    def rand(cls, *shape, low=0.0, high=1.0, dtype="float32",
        requires_grad=True):
        array = np.random.rand(*shape) * (high - low) + low
        return cls(array.astype(dtype), requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape, mean=0.0, std=1.0, dtype="float32",
         requires_grad=True):
        array = np.random.randn(*shape) * std + mean
        return cls(array.astype(dtype), requires_grad=requires_grad)

    @classmethod
    def constant(cls, *shape, c=1.0, dtype="float32", requires_grad=True):
        array = np.ones(shape) * c
        return cls(array.astype(dtype), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, dtype="float32", requires_grad=True):
        return cls.constant(*shape, c=1.0, dtype=dtype,
        requires_grad=requires_grad)

    @classmethod
    def zeros(cls, *shape, dtype="float32", requires_grad=True):
        return cls.constant(*shape, c=0.0, dtype=dtype,
         requires_grad=requires_grad)

    @classmethod
    def randb(cls, *shape, p=0.5, dtype="float32", requires_grad=True):
        array = np.random.rand(*shape) <= p
        return cls(array, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def empty(cls, *shape, dtype="float32", requires_grad=True):
        array = np.empty(shape, dtype=dtype)
        return cls(array, requires_grad=requires_grad)

    @classmethod
    def one_hot(cls, indices, num_classes, device=None, dtype="float32",
         requires_grad=True):
        one_hot_array = np.eye(num_classes, dtype=dtype)[np.array(
            indices.data, dtype=int)]
        return cls(one_hot_array, device=device, dtype=dtype,
            requires_grad=requires_grad)
