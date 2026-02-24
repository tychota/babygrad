import numpy as np
import pytest

from babygrad.tensor import Tensor
from babygrad.nn import (
    Parameter, Module, ReLU, Tanh, Sigmoid, Flatten, Linear,
    Sequential, Residual, Dropout, LayerNorm1d, BatchNorm1d,
    MSELoss,
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


# ── Residual ────────────────────────────────────────────────────────


class TestResidual:
    def test_residual_adds_input(self):
        """Residual computes fn(x) + x."""

        class DoubleIt(Module):
            def forward(self, x):
                return x * 2

        res = Residual(DoubleIt())
        x = Tensor([1.0, 2.0, 3.0])
        y = res(x)
        # fn(x) + x = 2x + x = 3x
        np.testing.assert_array_almost_equal(y.data, [3.0, 6.0, 9.0])

    def test_residual_is_module(self):
        class Identity(Module):
            def forward(self, x):
                return x

        assert isinstance(Residual(Identity()), Module)

    def test_residual_finds_inner_parameters(self):
        res = Residual(Linear(3, 3))
        params = res.parameters()
        assert len(params) == 2  # weight + bias from inner Linear


# ── Dropout ─────────────────────────────────────────────────────────


class TestDropout:
    def test_dropout_eval_returns_input(self):
        """In eval mode, Dropout is a no-op."""
        layer = Dropout(p=0.5)
        layer.eval()
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        y = layer(x)
        np.testing.assert_array_equal(y.data, x.data)

    def test_dropout_train_scales_output(self):
        """In train mode, non-dropped values are scaled by 1/(1-p)."""
        np.random.seed(42)
        layer = Dropout(p=0.5)
        x = Tensor(np.ones(1000, dtype=np.float32))
        y = layer(x)
        # Each surviving element should be 1/(1-0.5) = 2.0
        nonzero = y.data[y.data != 0.0]
        np.testing.assert_array_almost_equal(nonzero, np.full_like(nonzero, 2.0))

    def test_dropout_train_zeros_some_elements(self):
        """In train mode, roughly p fraction of elements are zeroed."""
        np.random.seed(42)
        layer = Dropout(p=0.5)
        x = Tensor(np.ones(10000, dtype=np.float32))
        y = layer(x)
        zero_fraction = np.mean(y.data == 0.0)
        # Should be roughly 0.5 (within tolerance)
        assert 0.4 < zero_fraction < 0.6

    def test_dropout_no_parameters(self):
        assert len(Dropout().parameters()) == 0

    def test_dropout_is_module(self):
        assert isinstance(Dropout(), Module)

    def test_dropout_preserves_mean(self):
        """Inverted dropout: E[output] ≈ E[input] during training."""
        np.random.seed(123)
        layer = Dropout(p=0.3)
        x = Tensor(np.ones(50000, dtype=np.float32) * 5.0)
        y = layer(x)
        # Mean should be close to 5.0 due to 1/(1-p) scaling
        assert abs(np.mean(y.data) - 5.0) < 0.2


# ── LayerNorm1d ─────────────────────────────────────────────────────


class TestLayerNorm1d:
    def test_output_shape(self):
        ln = LayerNorm1d(4)
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        y = ln(x)
        assert y.shape == (2, 4)

    def test_normalizes_to_zero_mean_unit_var(self):
        """With default weight=1 and bias=0, output has ~zero mean, ~unit var per sample."""
        ln = LayerNorm1d(8)
        x = Tensor(np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                              [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
                    requires_grad=True)
        y = ln(x)
        # Each row should have mean ≈ 0 and std ≈ 1
        for i in range(2):
            row = y.data[i]
            assert abs(np.mean(row)) < 1e-5
            assert abs(np.std(row) - 1.0) < 0.05

    def test_matches_pytorch_formula(self):
        """Verify against manual LayerNorm computation."""
        dim = 4
        ln = LayerNorm1d(dim)
        x_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        x = Tensor(x_np, requires_grad=True)
        y = ln(x)

        # Manual computation: mean=2.5, var=1.25
        mean = np.mean(x_np, axis=-1, keepdims=True)  # [[2.5]]
        var = np.var(x_np, axis=-1, keepdims=True)      # [[1.25]]
        expected = (x_np - mean) / np.sqrt(var + 1e-5)
        # weight=1, bias=0 by default
        np.testing.assert_array_almost_equal(y.data, expected, decimal=4)

    def test_has_weight_and_bias_parameters(self):
        ln = LayerNorm1d(5)
        params = ln.parameters()
        assert len(params) == 2
        assert isinstance(ln.weight, Parameter)
        assert isinstance(ln.bias, Parameter)

    def test_weight_and_bias_shape(self):
        ln = LayerNorm1d(6)
        assert ln.weight.shape == (6,)
        assert ln.bias.shape == (6,)

    def test_weight_initialized_to_ones(self):
        ln = LayerNorm1d(3)
        np.testing.assert_array_equal(ln.weight.data, [1.0, 1.0, 1.0])

    def test_bias_initialized_to_zeros(self):
        ln = LayerNorm1d(3)
        np.testing.assert_array_equal(ln.bias.data, [0.0, 0.0, 0.0])

    def test_is_module(self):
        assert isinstance(LayerNorm1d(4), Module)

    def test_backward_runs(self):
        """Verify gradients flow through LayerNorm."""
        ln = LayerNorm1d(4)
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        y = ln(x)
        loss = y * Tensor(np.ones_like(y.data))
        loss.backward(Tensor(np.ones_like(loss.data)))
        assert x.grad is not None
        assert x.grad.shape == (2, 4)


# ── BatchNorm1d ────────────────────────────────────────────────────


class TestBatchNorm1d:
    def test_output_shape(self):
        bn = BatchNorm1d(4)
        x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        y = bn(x)
        assert y.shape == (3, 4)

    def test_normalizes_over_batch_dimension(self):
        """In training mode, output has ~zero mean, ~unit var over batch dim."""
        bn = BatchNorm1d(4)
        x = Tensor(np.array([
            [10.0, 20.0, 30.0, 40.0],
            [12.0, 22.0, 32.0, 42.0],
            [14.0, 24.0, 34.0, 44.0],
            [16.0, 26.0, 36.0, 46.0],
        ], dtype=np.float32), requires_grad=True)
        y = bn(x)
        # Each column (feature) should have mean ≈ 0 and std ≈ 1
        for j in range(4):
            col = y.data[:, j]
            assert abs(np.mean(col)) < 1e-5
            assert abs(np.std(col) - 1.0) < 0.15

    def test_matches_formula(self):
        """Verify against manual BatchNorm computation."""
        bn = BatchNorm1d(2)
        x_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        x = Tensor(x_np, requires_grad=True)
        y = bn(x)

        # Manual: mean over batch axis=0, var over batch axis=0
        mean = np.mean(x_np, axis=0, keepdims=True)  # [[3, 4]]
        var = np.var(x_np, axis=0, keepdims=True)      # [[2.667, 2.667]]
        expected = (x_np - mean) / np.sqrt(var + 1e-5)
        # weight=1, bias=0 by default
        np.testing.assert_array_almost_equal(y.data, expected, decimal=4)

    def test_has_weight_and_bias_parameters(self):
        bn = BatchNorm1d(5)
        params = bn.parameters()
        assert len(params) == 2
        assert isinstance(bn.weight, Parameter)
        assert isinstance(bn.bias, Parameter)

    def test_running_mean_and_var_are_not_parameters(self):
        """running_mean and running_var are buffers, not Parameters."""
        bn = BatchNorm1d(4)
        params = bn.parameters()
        assert len(params) == 2  # only weight and bias
        assert not isinstance(bn.running_mean, Parameter)
        assert not isinstance(bn.running_var, Parameter)

    def test_weight_and_bias_shape(self):
        bn = BatchNorm1d(6)
        assert bn.weight.shape == (6,)
        assert bn.bias.shape == (6,)

    def test_weight_initialized_to_ones(self):
        bn = BatchNorm1d(3)
        np.testing.assert_array_equal(bn.weight.data, [1.0, 1.0, 1.0])

    def test_bias_initialized_to_zeros(self):
        bn = BatchNorm1d(3)
        np.testing.assert_array_equal(bn.bias.data, [0.0, 0.0, 0.0])

    def test_running_mean_initialized_to_zeros(self):
        bn = BatchNorm1d(3)
        np.testing.assert_array_equal(bn.running_mean.data, [0.0, 0.0, 0.0])

    def test_running_var_initialized_to_ones(self):
        bn = BatchNorm1d(3)
        np.testing.assert_array_equal(bn.running_var.data, [1.0, 1.0, 1.0])

    def test_running_stats_updated_during_training(self):
        """running_mean and running_var update after forward in training mode."""
        bn = BatchNorm1d(2, momentum=0.1)
        x_np = np.array([[1.0, 10.0], [5.0, 20.0], [9.0, 30.0]], dtype=np.float32)
        x = Tensor(x_np)
        bn(x)
        # After one batch: running_mean = (1-0.1)*0 + 0.1*batch_mean
        batch_mean = np.mean(x_np, axis=0)
        batch_var = np.var(x_np, axis=0)
        expected_running_mean = 0.9 * 0.0 + 0.1 * batch_mean
        expected_running_var = 0.9 * 1.0 + 0.1 * batch_var
        np.testing.assert_array_almost_equal(bn.running_mean.data, expected_running_mean)
        np.testing.assert_array_almost_equal(bn.running_var.data, expected_running_var)

    def test_eval_uses_running_stats(self):
        """In eval mode, uses running_mean/running_var instead of batch stats."""
        bn = BatchNorm1d(2)
        # Train on a batch to build up running stats
        x_train = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
        bn(x_train)
        saved_mean = bn.running_mean.data.copy()
        saved_var = bn.running_var.data.copy()

        bn.eval()
        # In eval, even a single sample should work (no batch stats needed)
        x_eval = Tensor(np.array([[2.0, 3.0]], dtype=np.float32))
        y = bn(x_eval)
        # Manual: (x - running_mean) / sqrt(running_var + eps) * weight + bias
        expected = (np.array([[2.0, 3.0]]) - saved_mean) / np.sqrt(saved_var + 1e-5)
        np.testing.assert_array_almost_equal(y.data, expected, decimal=4)

    def test_is_module(self):
        assert isinstance(BatchNorm1d(4), Module)

    def test_backward_runs(self):
        """Verify gradients flow through BatchNorm."""
        bn = BatchNorm1d(4)
        x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        y = bn(x)
        loss = y * Tensor(np.ones_like(y.data))
        loss.backward(Tensor(np.ones_like(loss.data)))
        assert x.grad is not None
        assert x.grad.shape == (3, 4)


# ── MSELoss ────────────────────────────────────────────────────────


class TestMSELoss:
    def test_mse_perfect_prediction(self):
        """MSE is 0 when pred == target."""
        loss_fn = MSELoss()
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.0, 2.0, 3.0])
        loss = loss_fn(pred, target)
        np.testing.assert_array_almost_equal(loss.data, 0.0)

    def test_mse_known_value(self):
        """MSE matches manual computation."""
        loss_fn = MSELoss()
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([2.0, 2.0, 2.0])
        loss = loss_fn(pred, target)
        # diff = [-1, 0, 1], sq = [1, 0, 1], mean = 2/3
        np.testing.assert_array_almost_equal(loss.data, 2.0 / 3.0, decimal=5)

    def test_mse_2d(self):
        """MSE works on 2D tensors."""
        loss_fn = MSELoss()
        pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = Tensor([[1.0, 1.0], [1.0, 1.0]])
        loss = loss_fn(pred, target)
        # diff = [[0,1],[2,3]], sq = [[0,1],[4,9]], sum=14, mean=14/4=3.5
        np.testing.assert_array_almost_equal(loss.data, 3.5)

    def test_mse_backward(self):
        """Gradients flow through MSELoss."""
        loss_fn = MSELoss()
        pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = Tensor([2.0, 2.0, 2.0])
        loss = loss_fn(pred, target)
        loss.backward()
        # d(MSE)/d(pred) = 2*(pred - target)/n
        expected_grad = 2.0 * (np.array([1.0, 2.0, 3.0]) - np.array([2.0, 2.0, 2.0])) / 3.0
        np.testing.assert_array_almost_equal(pred.grad, expected_grad, decimal=5)

    def test_mse_is_module(self):
        assert isinstance(MSELoss(), Module)

    def test_mse_no_parameters(self):
        assert len(MSELoss().parameters()) == 0
