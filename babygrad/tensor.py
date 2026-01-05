import numbers

import numpy as np

NDArray = np.ndarray


class Tensor:
    def __init__(self, data, *, device=None, dtype="float32",
         requires_grad=True):
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

        self.requires_grad = requires_grad
        self._op = None
        self._inputs = []

        if isinstance(data, Tensor):
            self.data = data.data
        elif isinstance(data, NDArray):
            self.data = data
        elif isinstance(data, list) or isinstance(data, numbers.Number):
            self.data = np.asarray(data)
    
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
