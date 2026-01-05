import numpy as np
import pytest

from babygrad.tensor import Tensor


class TestTensorInit:
    """Tests for Tensor initialization with different data types."""

    def test_from_list(self):
        """Tensor can be created from a Python list."""
        t = Tensor([1, 2, 3])
        assert isinstance(t.data, np.ndarray)
        np.testing.assert_array_equal(t.data, [1, 2, 3])

    def test_from_nested_list(self):
        """Tensor can be created from a nested list (2D array)."""
        t = Tensor([[1, 2], [3, 4]])
        assert isinstance(t.data, np.ndarray)
        assert t.data.shape == (2, 2)
        np.testing.assert_array_equal(t.data, [[1, 2], [3, 4]])

    def test_from_scalar_int(self):
        """Tensor can be created from an integer scalar."""
        t = Tensor(42)
        assert isinstance(t.data, np.ndarray)
        assert t.data == 42

    def test_from_scalar_float(self):
        """Tensor can be created from a float scalar."""
        t = Tensor(3.14)
        assert isinstance(t.data, np.ndarray)
        assert t.data == pytest.approx(3.14)

    def test_from_ndarray(self):
        """Tensor can be created from a numpy ndarray."""
        arr = np.array([1.0, 2.0, 3.0])
        t = Tensor(arr)
        assert isinstance(t.data, np.ndarray)
        np.testing.assert_array_equal(t.data, arr)

    def test_from_tensor(self):
        """Tensor can be created from another Tensor."""
        t1 = Tensor([1, 2, 3])
        t2 = Tensor(t1)
        assert isinstance(t2.data, np.ndarray)
        np.testing.assert_array_equal(t2.data, t1.data)

    def test_from_tensor_shares_data(self):
        """Tensor created from another Tensor shares the same data array."""
        t1 = Tensor([1, 2, 3])
        t2 = Tensor(t1)
        # They share the same underlying array
        assert t2.data is t1.data


class TestTensorAttributes:
    """Tests for Tensor attributes."""

    def test_requires_grad_default_true(self):
        """requires_grad defaults to True."""
        t = Tensor([1, 2, 3])
        assert t.requires_grad is True

    def test_requires_grad_can_be_false(self):
        """requires_grad can be set to False."""
        t = Tensor([1, 2, 3], requires_grad=False)
        assert t.requires_grad is False

    def test_op_is_none(self):
        """_op is None for newly created tensors."""
        t = Tensor([1, 2, 3])
        assert t._op is None

    def test_inputs_is_empty_list(self):
        """_inputs is an empty list for newly created tensors."""
        t = Tensor([1, 2, 3])
        assert t._inputs == []
        assert isinstance(t._inputs, list)


class TestTensorRepr:
    """Tests for Tensor string representations."""

    def test_repr_with_requires_grad_true(self):
        """__repr__ shows data and requires_grad=True."""
        t = Tensor([1, 2, 3])
        result = repr(t)
        assert "Tensor(" in result
        assert "requires_grad=True" in result

    def test_repr_with_requires_grad_false(self):
        """__repr__ shows data and requires_grad=False."""
        t = Tensor([1, 2, 3], requires_grad=False)
        result = repr(t)
        assert "Tensor(" in result
        assert "requires_grad=False" in result

    def test_str_shows_data_only(self):
        """__str__ shows just the data array."""
        t = Tensor([1, 2, 3])
        result = str(t)
        # Should not contain Tensor or requires_grad
        assert "Tensor" not in result
        assert "requires_grad" not in result
