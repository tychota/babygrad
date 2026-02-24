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

    def test_from_tensor_copies_data(self):
        """Tensor created from another Tensor copies the data (via numpy())."""
        t1 = Tensor([1, 2, 3])
        t2 = Tensor(t1)
        # They do NOT share the same underlying array (numpy() returns copy)
        assert t2.data is not t1.data
        np.testing.assert_array_equal(t2.data, t1.data)

    def test_dtype_conversion(self):
        """Tensor converts data to specified dtype."""
        t = Tensor([1, 2, 3], dtype="float64")
        assert t.data.dtype == np.float64

    def test_dtype_default_float32(self):
        """Tensor defaults to float32 dtype."""
        t = Tensor([1, 2, 3])
        assert t.data.dtype == np.float32

    def test_dtype_from_ndarray_preserved(self):
        """When passing ndarray, dtype is converted to specified dtype."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        t = Tensor(arr, dtype="float32")
        assert t.data.dtype == np.float32

    def test_device_default_cpu(self):
        """Device defaults to 'cpu'."""
        t = Tensor([1, 2, 3])
        assert t.device == "cpu"

    def test_device_custom(self):
        """Device can be set to custom value."""
        t = Tensor([1, 2, 3], device="cuda")
        assert t.device == "cuda"


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

    def test_grad_is_none(self):
        """grad is None for newly created tensors."""
        t = Tensor([1, 2, 3])
        assert t.grad is None


class TestTensorProperties:
    """Tests for Tensor properties."""

    def test_shape(self):
        """shape property returns correct shape."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t.shape == (2, 3)

    def test_dtype(self):
        """dtype property returns correct dtype."""
        t = Tensor([1, 2, 3], dtype="float64")
        assert t.dtype == np.float64

    def test_ndim(self):
        """ndim property returns number of dimensions."""
        t = Tensor([[1, 2], [3, 4]])
        assert t.ndim == 2

    def test_size(self):
        """size property returns total number of elements."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t.size == 6

    def test_op_property(self):
        """op property returns _op value."""
        t = Tensor([1, 2, 3])
        assert t.op is None

    def test_inputs_property(self):
        """inputs property returns _inputs value."""
        t = Tensor([1, 2, 3])
        assert t.inputs == []

    def test_device_property(self):
        """device property returns _device value."""
        t = Tensor([1, 2, 3])
        assert t.device == "cpu"


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


class TestTensorNumpy:
    """Tests for Tensor.numpy() method."""

    def test_numpy_returns_ndarray(self):
        """numpy() returns a numpy ndarray."""
        t = Tensor([1, 2, 3])
        result = t.numpy()
        assert isinstance(result, np.ndarray)

    def test_numpy_returns_copy(self):
        """numpy() returns a copy, not the original data."""
        t = Tensor([1, 2, 3])
        result = t.numpy()
        assert result is not t.data

    def test_numpy_modification_does_not_affect_tensor(self):
        """Modifying numpy() result does not affect the tensor."""
        t = Tensor([1, 2, 3])
        result = t.numpy()
        result[0] = 999
        assert t.data[0] != 999

    def test_numpy_values_match(self):
        """numpy() returns array with same values."""
        t = Tensor([1, 2, 3])
        np.testing.assert_array_equal(t.numpy(), t.data)


class TestTensorDetach:
    """Tests for Tensor.detach() method."""

    def test_detach_returns_tensor(self):
        """detach() returns a Tensor."""
        t = Tensor([1, 2, 3], requires_grad=True)
        result = t.detach()
        assert isinstance(result, Tensor)

    def test_detach_requires_grad_false(self):
        """detach() returns tensor with requires_grad=False."""
        t = Tensor([1, 2, 3], requires_grad=True)
        result = t.detach()
        assert result.requires_grad is False

    def test_detach_same_values(self):
        """detach() returns tensor with same data values."""
        t = Tensor([1, 2, 3])
        result = t.detach()
        np.testing.assert_array_equal(result.data, t.data)

    def test_detach_is_new_tensor(self):
        """detach() returns a new tensor, not the same object."""
        t = Tensor([1, 2, 3])
        result = t.detach()
        assert result is not t


class TestTensorStaticMethods:
    """Tests for Tensor static factory methods."""

    def test_randn_shape(self):
        t = Tensor.randn(3, 4)
        assert t.shape == (3, 4)
        assert t.dtype == np.float32

    def test_zeros_shape(self):
        t = Tensor.zeros(2, 3)
        assert t.shape == (2, 3)
        np.testing.assert_array_equal(t.data, np.zeros((2, 3)))

    def test_ones_shape(self):
        t = Tensor.ones(2, 3)
        assert t.shape == (2, 3)
        np.testing.assert_array_equal(t.data, np.ones((2, 3)))

    def test_randb_shape_and_values(self):
        np.random.seed(0)
        t = Tensor.randb(100, p=0.5)
        assert t.shape == (100,)
        # All values should be 0 or 1
        assert set(np.unique(t.data)).issubset({0.0, 1.0})


class TestTensorInstanceMethods:
    """Tests for reshape and broadcast_to instance methods."""

    def test_reshape(self):
        t = Tensor(np.arange(6, dtype=np.float32))
        r = t.reshape(2, 3)
        assert r.shape == (2, 3)

    def test_broadcast_to(self):
        t = Tensor(np.array([[1.0], [2.0]]))
        b = t.broadcast_to((2, 3))
        assert b.shape == (2, 3)
        np.testing.assert_array_equal(b.data, [[1, 1, 1], [2, 2, 2]])
