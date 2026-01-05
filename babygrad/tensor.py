import numpy as np

NDArray = np.ndarray


class Tensor:
    def __init__(self, data, *, device=None, dtype="float32", requires_grad=True):
        """
        Create a new tensor.
        Args:
            data: Array-like data (list, numpy array, or another Tensor)
            device: Device placement (currently ignored, CPU only)
            dtype: Data type for the array
            requires_grad: Whether to track gradients for this tensor

        Design decision: requires_grad defaults to True (unlike PyTorch)
         (Will change later to false, when introducing Parameter)
        """
        if isinstance(data, Tensor):
            if dtype is None:
                dtype = data.dtype
            self.data = data.numpy().astype(dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype if dtype is not None else data.dtype)
        else:
            self.data = np.array(data, dtype=dtype if dtype is not None else "float32")

        self.grad = None
        self.requires_grad = requires_grad
        self._op = None
        self._inputs = []
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
        # we will do this in next chapter !.
        pass

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
