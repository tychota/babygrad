import numpy as np
import pytest

from babygrad.tensor import Tensor
from babygrad.nn import (
    Parameter, Module, ReLU, Tanh, Sigmoid, Flatten, Linear,
    Sequential,
)


class TestParameter:
    """Tests for the Parameter class."""

    def test_parameter_is_tensor(self):
        """Parameter is a subclass of Tensor."""
        p = Parameter(Tensor([1.0, 2.0, 3.0]))
        assert isinstance(p, Tensor)

    def test_parameter_requires_grad_true(self):
        """Parameter always has requires_grad=True."""
        p = Parameter(Tensor([1.0, 2.0, 3.0]))
        assert p.requires_grad is True

    def test_parameter_from_tensor_without_grad(self):
        """Parameter from a non-grad tensor still has requires_grad=True."""
        t = Tensor([1.0, 2.0], requires_grad=False)
        p = Parameter(t)
        assert p.requires_grad is True

    def test_parameter_data_values(self):
        """Parameter preserves the data values."""
        p = Parameter(Tensor([4.0, 5.0, 6.0]))
        np.testing.assert_array_almost_equal(p.data, [4.0, 5.0, 6.0])

    def test_parameter_supports_backward(self):
        """Parameter can participate in backward pass."""
        p = Parameter(Tensor([2.0, 3.0]))
        y = p * Tensor([4.0, 5.0], requires_grad=True)
        y.backward(Tensor([1.0, 1.0]))
        np.testing.assert_array_almost_equal(p.grad, [4.0, 5.0])


class TestModuleParameters:
    """Tests for Module.parameters() discovery."""

    def test_finds_direct_parameters(self):
        """parameters() finds Parameters stored as direct attributes."""

        class M(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter(Tensor([1.0, 2.0]))
                self.b = Parameter(Tensor([0.0]))

            def forward(self, x):
                return x

        m = M()
        params = m.parameters()
        assert len(params) == 2
        assert m.w in params
        assert m.b in params

    def test_ignores_plain_tensors(self):
        """parameters() does not include plain Tensors."""

        class M(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(Tensor([1.0, 2.0]))
                self.data = Tensor([3.0, 4.0])  # not a Parameter

            def forward(self, x):
                return x

        m = M()
        params = m.parameters()
        assert len(params) == 1
        assert m.weight in params

    def test_finds_parameters_in_list(self):
        """parameters() finds Parameters nested in a list."""

        class M(Module):
            def __init__(self):
                super().__init__()
                self.params_list = [
                    Parameter(Tensor([1.0])),
                    Parameter(Tensor([2.0])),
                ]

            def forward(self, x):
                return x

        m = M()
        params = m.parameters()
        assert len(params) == 2

    def test_finds_parameters_in_dict(self):
        """parameters() finds Parameters nested in a dict."""

        class M(Module):
            def __init__(self):
                super().__init__()
                self.params_dict = {
                    "a": Parameter(Tensor([1.0])),
                    "b": Parameter(Tensor([2.0])),
                }

            def forward(self, x):
                return x

        m = M()
        params = m.parameters()
        assert len(params) == 2

    def test_finds_parameters_in_child_modules(self):
        """parameters() finds Parameters inside nested Modules."""

        class Child(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter(Tensor([1.0]))

            def forward(self, x):
                return x

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.child = Child()
                self.own = Parameter(Tensor([2.0]))

            def forward(self, x):
                return x

        m = Parent()
        params = m.parameters()
        assert len(params) == 2

    def test_deduplicates_shared_parameters(self):
        """parameters() deduplicates shared Parameter references."""

        class M(Module):
            def __init__(self):
                super().__init__()
                shared = Parameter(Tensor([1.0]))
                self.a = shared
                self.b = shared

            def forward(self, x):
                return x

        m = M()
        params = m.parameters()
        assert len(params) == 1


class TestModuleCallable:
    """Tests for Module.__call__ delegating to forward."""

    def test_call_invokes_forward(self):
        """Calling a module invokes its forward method."""

        class M(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        m = M()
        t = Tensor([1.0, 2.0])
        result = m(t)
        np.testing.assert_array_equal(result.data, t.data)

    def test_forward_not_implemented(self):
        """Base Module.forward raises NotImplementedError."""

        class Empty(Module):
            def __init__(self):
                super().__init__()

        m = Empty()
        with pytest.raises(NotImplementedError):
            m(Tensor([1.0]))


class TestModuleTrainingState:
    """Tests for Module.train() and eval()."""

    def test_default_training_true(self):
        """Module starts in training mode."""

        class M(Module):
            def __init__(self):
                super().__init__()

        assert M().training is True

    def test_eval_sets_training_false(self):
        """eval() sets training to False."""

        class M(Module):
            def __init__(self):
                super().__init__()

        m = M()
        m.eval()
        assert m.training is False

    def test_train_sets_training_true(self):
        """train() sets training back to True after eval()."""

        class M(Module):
            def __init__(self):
                super().__init__()

        m = M()
        m.eval()
        m.train()
        assert m.training is True

    def test_eval_propagates_to_child_modules(self):
        """eval() sets training=False on child modules recursively."""

        class Child(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.children_list = [Child(), Child()]

            def forward(self, x):
                return x

        net = Parent()
        net.eval()
        assert net.training is False
        assert net.children_list[0].training is False
        assert net.children_list[1].training is False

    def test_train_propagates_to_child_modules(self):
        """train() sets training=True on child modules recursively."""

        class Child(Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.children_list = [Child(), Child()]

            def forward(self, x):
                return x

        net = Parent()
        net.eval()
        net.train()
        assert net.training is True
        assert net.children_list[0].training is True
        assert net.children_list[1].training is True


# ── Stateless layers ────────────────────────────────────────────────


class TestReLULayer:
    def test_relu_forward(self):
        layer = ReLU()
        x = Tensor([-1.0, 0.0, 2.0])
        y = layer(x)
        np.testing.assert_array_almost_equal(y.data, [0.0, 0.0, 2.0])

    def test_relu_is_module(self):
        assert isinstance(ReLU(), Module)

    def test_relu_no_parameters(self):
        assert len(ReLU().parameters()) == 0


class TestTanhLayer:
    def test_tanh_forward(self):
        layer = Tanh()
        x = Tensor([0.0, 1.0, -1.0])
        y = layer(x)
        np.testing.assert_array_almost_equal(y.data, np.tanh([0.0, 1.0, -1.0]), decimal=5)

    def test_tanh_is_module(self):
        assert isinstance(Tanh(), Module)


class TestSigmoidLayer:
    def test_sigmoid_forward(self):
        layer = Sigmoid()
        x = Tensor([0.0, 2.0, -2.0])
        y = layer(x)
        expected = 1.0 / (1.0 + np.exp(-np.array([0.0, 2.0, -2.0])))
        np.testing.assert_array_almost_equal(y.data, expected, decimal=5)

    def test_sigmoid_is_module(self):
        assert isinstance(Sigmoid(), Module)


class TestFlattenLayer:
    def test_flatten_3d(self):
        """Flatten (2,3,4) -> (2,12)."""
        layer = Flatten()
        x = Tensor(np.ones((2, 3, 4)))
        y = layer(x)
        assert y.shape == (2, 12)

    def test_flatten_4d(self):
        """Flatten (2,3,4,5) -> (2,60)."""
        layer = Flatten()
        x = Tensor(np.ones((2, 3, 4, 5)))
        y = layer(x)
        assert y.shape == (2, 60)

    def test_flatten_preserves_values(self):
        layer = Flatten()
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        x = Tensor(data)
        y = layer(x)
        np.testing.assert_array_equal(y.data, data.reshape(2, 12))

    def test_flatten_is_module(self):
        assert isinstance(Flatten(), Module)

    def test_flatten_no_parameters(self):
        assert len(Flatten().parameters()) == 0


# ── Linear layer ────────────────────────────────────────────────────


class TestLinearLayer:
    def test_linear_output_shape(self):
        layer = Linear(3, 5)
        x = Tensor(np.ones((2, 3), dtype=np.float32))
        y = layer(x)
        assert y.shape == (2, 5)

    def test_linear_has_weight_and_bias_parameters(self):
        layer = Linear(4, 3)
        params = layer.parameters()
        assert len(params) == 2
        assert layer.weight.shape == (4, 3)
        assert layer.bias.shape == (1, 3)

    def test_linear_weight_is_parameter(self):
        layer = Linear(4, 3)
        assert isinstance(layer.weight, Parameter)
        assert isinstance(layer.bias, Parameter)

    def test_linear_no_bias(self):
        layer = Linear(4, 3, bias=False)
        params = layer.parameters()
        assert len(params) == 1
        assert layer.bias is None

    def test_linear_no_bias_output_shape(self):
        layer = Linear(3, 5, bias=False)
        x = Tensor(np.ones((2, 3), dtype=np.float32))
        y = layer(x)
        assert y.shape == (2, 5)

    def test_linear_forward_computes_xW_plus_b(self):
        """Verify y = x @ W + b with known values."""
        layer = Linear(2, 3)
        # Set known weights
        layer.weight = Parameter(Tensor(np.ones((2, 3), dtype=np.float32)))
        layer.bias = Parameter(Tensor(np.array([0.1, 0.2, 0.3], dtype=np.float32)))
        x = Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
        y = layer(x)
        # x @ W = [1,2] @ [[1,1,1],[1,1,1]] = [3,3,3]
        # + bias = [3.1, 3.2, 3.3]
        np.testing.assert_array_almost_equal(y.data, [[3.1, 3.2, 3.3]])


# ── Sequential ──────────────────────────────────────────────────────


class TestSequential:
    def test_sequential_chains_layers(self):
        model = Sequential(Linear(3, 5), ReLU(), Linear(5, 2))
        x = Tensor(np.ones((1, 3), dtype=np.float32))
        y = model(x)
        assert y.shape == (1, 2)

    def test_sequential_is_module(self):
        assert isinstance(Sequential(), Module)

    def test_sequential_finds_all_parameters(self):
        model = Sequential(Linear(3, 5), ReLU(), Linear(5, 2))
        params = model.parameters()
        # Linear(3,5): weight + bias = 2, Linear(5,2): weight + bias = 2
        assert len(params) == 4

    def test_sequential_eval_propagates(self):
        model = Sequential(Linear(3, 5), ReLU(), Linear(5, 2))
        model.eval()
        assert model.training is False
