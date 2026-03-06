import numpy as np
import pytest

from babygrad.tensor import Tensor
from babygrad.nn import (
    Parameter, Module, ReLU, Tanh, Sigmoid, GELU, SiLU, Flatten, Linear,
    Sequential, Residual, Dropout, LayerNorm1d, BatchNorm1d,
    MSELoss, SoftmaxLoss, CrossEntropyLoss,
    Embedding, RMSNorm, SwiGLU, MultiHeadAttention, RotaryPositionEmbedding,
    GroupedQueryAttention, TransformerBlock, Transformer,
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


class TestGELUModule:
    def test_forward(self):
        from babygrad.ops import gelu
        layer = GELU()
        x = Tensor([1.0, 0.0, -1.0])
        result = layer(x)
        expected = gelu(x)
        np.testing.assert_allclose(result.data, expected.data)

    def test_is_module(self):
        assert isinstance(GELU(), Module)

    def test_no_parameters(self):
        assert len(GELU().parameters()) == 0


class TestSiLUModule:
    def test_forward(self):
        from babygrad.ops import silu
        layer = SiLU()
        x = Tensor([1.0, 0.0, -1.0])
        result = layer(x)
        expected = silu(x)
        np.testing.assert_allclose(result.data, expected.data)

    def test_is_module(self):
        assert isinstance(SiLU(), Module)


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


# ── SoftmaxLoss ────────────────────────────────────────────────────


class TestSoftmaxLoss:
    def test_softmax_loss_known_value(self):
        """Dog example from spec: logits=[2,5,0.1], label=1 (Dog)."""
        loss_fn = SoftmaxLoss()
        logits = Tensor(np.array([[2.0, 5.0, 0.1]], dtype=np.float32))
        y = np.array([1])  # Dog
        loss = loss_fn(logits, y)
        # logsumexp([2,5,0.1]) - 5.0 (the Dog logit)
        lse = np.log(np.sum(np.exp([2.0, 5.0, 0.1])))
        expected = (lse - 5.0) / 1.0
        np.testing.assert_array_almost_equal(loss.data, expected, decimal=4)

    def test_softmax_loss_batch(self):
        """Batch of 2 samples."""
        loss_fn = SoftmaxLoss()
        logits = Tensor(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32))
        y = np.array([2, 0])
        loss = loss_fn(logits, y)
        # Sample 0: logsumexp([1,2,3]) - 3, Sample 1: logsumexp([1,2,3]) - 1
        lse = np.log(np.sum(np.exp([1.0, 2.0, 3.0])))
        expected = ((lse - 3.0) + (lse - 1.0)) / 2.0
        np.testing.assert_array_almost_equal(loss.data, expected, decimal=4)

    def test_softmax_loss_perfect_prediction(self):
        """Loss is small when model is very confident and correct."""
        loss_fn = SoftmaxLoss()
        logits = Tensor(np.array([[0.0, 100.0, 0.0]], dtype=np.float32))
        y = np.array([1])
        loss = loss_fn(logits, y)
        assert loss.data < 0.01

    def test_softmax_loss_backward(self):
        """Gradients flow through SoftmaxLoss."""
        loss_fn = SoftmaxLoss()
        logits = Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), requires_grad=True)
        y = np.array([1])
        loss = loss_fn(logits, y)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == (1, 3)

    def test_softmax_loss_gradient_is_softmax_minus_onehot(self):
        """d(loss)/d(logits) = (softmax(logits) - one_hot(y)) / n."""
        loss_fn = SoftmaxLoss()
        logits_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        logits = Tensor(logits_np, requires_grad=True)
        y = np.array([1])
        loss = loss_fn(logits, y)
        loss.backward()
        # softmax
        exp_l = np.exp(logits_np)
        sm = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        one_hot = np.array([[0, 1, 0]], dtype=np.float32)
        expected = (sm - one_hot) / 1.0
        np.testing.assert_array_almost_equal(logits.grad, expected, decimal=4)

    def test_softmax_loss_is_module(self):
        assert isinstance(SoftmaxLoss(), Module)

    def test_softmax_loss_no_parameters(self):
        assert len(SoftmaxLoss().parameters()) == 0


# ── CrossEntropyLoss ───────────────────────────────────────────────


class TestCrossEntropyLoss:
    def test_2d_logits_matches_softmax_loss(self):
        """(B, C) logits + (B,) targets — same as SoftmaxLoss."""
        ce = CrossEntropyLoss()
        sf = SoftmaxLoss()
        logits = Tensor(np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]], dtype=np.float32))
        y = np.array([2, 0])
        np.testing.assert_array_almost_equal(ce(logits, y).data, sf(logits, y).data, decimal=5)

    def test_3d_logits_language_model(self):
        """(B, L, vocab) logits + (B, L) targets for language modelling."""
        ce = CrossEntropyLoss()
        # B=2, L=3, vocab=4
        logits_np = np.random.randn(2, 3, 4).astype(np.float32)
        logits = Tensor(logits_np)
        targets = np.array([[0, 1, 2], [3, 0, 1]])
        loss = ce(logits, targets)
        # Manual: reshape to (6,4) and (6,), compute SoftmaxLoss
        flat_logits = logits_np.reshape(-1, 4)
        flat_targets = targets.reshape(-1)
        sf = SoftmaxLoss()
        expected = sf(Tensor(flat_logits), flat_targets)
        np.testing.assert_array_almost_equal(loss.data, expected.data, decimal=4)

    def test_ignore_index(self):
        """Tokens with ignore_index should not contribute to loss."""
        ce = CrossEntropyLoss(ignore_index=-1)
        logits = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        targets = np.array([2, -1])  # second sample is padding
        loss = ce(logits, targets)
        # Only first sample contributes
        sf = SoftmaxLoss()
        single_loss = sf(Tensor(np.array([[1.0, 2.0, 3.0]], dtype=np.float32)), np.array([2]))
        np.testing.assert_array_almost_equal(loss.data, single_loss.data, decimal=5)

    def test_ignore_index_3d(self):
        """ignore_index works with (B, L, vocab) inputs."""
        ce = CrossEntropyLoss(ignore_index=0)
        # B=1, L=3, vocab=3
        logits_np = np.array([[[1.0, 2.0, 3.0],
                               [2.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0]]], dtype=np.float32)
        logits = Tensor(logits_np)
        targets = np.array([[2, 1, 0]])  # last token is pad (ignore_index=0)
        loss = ce(logits, targets)
        # Only first two positions contribute
        flat_logits = logits_np[0, :2, :]  # (2, 3)
        flat_targets = np.array([2, 1])
        sf = SoftmaxLoss()
        expected = sf(Tensor(flat_logits), flat_targets)
        np.testing.assert_array_almost_equal(loss.data, expected.data, decimal=4)

    def test_backward(self):
        """Gradients flow through CrossEntropyLoss."""
        ce = CrossEntropyLoss()
        logits = Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
        targets = np.array([[0, 1, 2], [3, 0, 1]])
        loss = ce(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == (2, 3, 4)

    def test_is_module(self):
        assert isinstance(CrossEntropyLoss(), Module)

    def test_no_parameters(self):
        assert len(CrossEntropyLoss().parameters()) == 0


class TestEmbeddingModule:
    def test_output_shape(self):
        emb = Embedding(vocab_size=100, embed_dim=32)
        x = Tensor([0, 5, 10])
        result = emb(x)
        assert result.shape == (3, 32)

    def test_2d_input(self):
        emb = Embedding(vocab_size=100, embed_dim=32)
        x = Tensor([[0, 1], [2, 3]])
        result = emb(x)
        assert result.shape == (2, 2, 32)

    def test_has_weight_parameter(self):
        emb = Embedding(vocab_size=10, embed_dim=5)
        params = emb.parameters()
        assert len(params) == 1
        assert params[0].shape == (10, 5)

    def test_is_module(self):
        assert isinstance(Embedding(10, 5), Module)

    def test_backward(self):
        from babygrad.ops import summation
        emb = Embedding(vocab_size=5, embed_dim=3)
        x = Tensor([0, 1, 0])
        result = emb(x)
        loss = summation(result)
        loss.backward()
        assert emb.weight.grad is not None


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=4)
        x = Tensor(np.random.randn(2, 4).astype(np.float32))
        result = norm(x)
        assert result.shape == (2, 4)

    def test_rms_normalized(self):
        norm = RMSNorm(dim=4, eps=1e-5)
        x = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        result = norm(x)
        rms = np.sqrt(np.mean(result.data ** 2, axis=-1))
        np.testing.assert_allclose(rms, [1.0], atol=1e-4)

    def test_has_weight_parameter(self):
        norm = RMSNorm(dim=8)
        params = norm.parameters()
        assert len(params) == 1
        assert params[0].shape == (8,)

    def test_weight_initialized_to_ones(self):
        norm = RMSNorm(dim=4)
        np.testing.assert_array_equal(norm.weight.data, [1, 1, 1, 1])

    def test_is_module(self):
        assert isinstance(RMSNorm(4), Module)

    def test_backward(self):
        from babygrad.ops import summation
        norm = RMSNorm(dim=4)
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        result = norm(x)
        loss = summation(result)
        loss.backward()
        assert x.grad is not None


class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLU(dim=8, hidden_dim=16)
        x = Tensor(np.random.randn(2, 8).astype(np.float32))
        result = ffn(x)
        assert result.shape == (2, 8)

    def test_has_parameters(self):
        ffn = SwiGLU(dim=8, hidden_dim=16)
        params = ffn.parameters()
        assert len(params) == 3  # w1, w2, w3

    def test_is_module(self):
        assert isinstance(SwiGLU(8, 16), Module)

    def test_backward(self):
        ffn = SwiGLU(dim=4, hidden_dim=8)
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        result = ffn(x)
        result.sum().backward()
        assert x.grad is not None


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(embed_dim=32, num_heads=4)
        x = Tensor(np.random.randn(2, 10, 32).astype(np.float32))
        result = mha(x)
        assert result.shape == (2, 10, 32)

    def test_causal_mask(self):
        mha = MultiHeadAttention(embed_dim=8, num_heads=2, causal=True)
        x = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        result = mha(x)
        assert result.shape == (1, 4, 8)

    def test_has_parameters(self):
        mha = MultiHeadAttention(embed_dim=16, num_heads=4)
        params = mha.parameters()
        assert len(params) >= 4

    def test_is_module(self):
        assert isinstance(MultiHeadAttention(16, 4), Module)

    def test_backward(self):
        mha = MultiHeadAttention(embed_dim=8, num_heads=2)
        x = Tensor(np.random.randn(1, 3, 8).astype(np.float32), requires_grad=True)
        result = mha(x)
        result.sum().backward()
        assert x.grad is not None

    def test_dropout_in_eval(self):
        mha = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0.5)
        mha.eval()
        x = Tensor(np.random.randn(1, 3, 8).astype(np.float32))
        r1 = mha(x)
        r2 = mha(x)
        np.testing.assert_allclose(r1.data, r2.data)


class TestRotaryPositionEmbedding:
    def test_output_shape_unchanged(self):
        rope = RotaryPositionEmbedding(dim=8)
        q = Tensor(np.random.randn(1, 4, 2, 8).astype(np.float32))  # (B, H, L, D)
        result = rope(q, seq_len=2)
        assert result.shape == q.shape

    def test_different_positions_give_different_outputs(self):
        rope = RotaryPositionEmbedding(dim=8)
        q = Tensor(np.ones((1, 1, 3, 8), dtype=np.float32))
        result = rope(q, seq_len=3)
        assert not np.allclose(result.data[0, 0, 0], result.data[0, 0, 1])

    def test_is_module(self):
        assert isinstance(RotaryPositionEmbedding(8), Module)


class TestGroupedQueryAttention:
    def test_output_shape(self):
        gqa = GroupedQueryAttention(embed_dim=32, num_heads=8, num_kv_heads=2)
        x = Tensor(np.random.randn(2, 10, 32).astype(np.float32))
        result = gqa(x)
        assert result.shape == (2, 10, 32)

    def test_num_kv_heads_divides_num_heads(self):
        gqa = GroupedQueryAttention(embed_dim=16, num_heads=4, num_kv_heads=2)
        x = Tensor(np.random.randn(1, 5, 16).astype(np.float32))
        result = gqa(x)
        assert result.shape == (1, 5, 16)

    def test_has_parameters(self):
        gqa = GroupedQueryAttention(embed_dim=16, num_heads=4, num_kv_heads=2)
        params = gqa.parameters()
        assert len(params) >= 4

    def test_is_module(self):
        assert isinstance(GroupedQueryAttention(16, 4, 2), Module)

    def test_backward(self):
        gqa = GroupedQueryAttention(embed_dim=8, num_heads=4, num_kv_heads=2)
        x = Tensor(np.random.randn(1, 3, 8).astype(np.float32), requires_grad=True)
        result = gqa(x)
        result.sum().backward()
        assert x.grad is not None

    def test_causal(self):
        gqa = GroupedQueryAttention(embed_dim=8, num_heads=4, num_kv_heads=2, causal=True)
        x = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        result = gqa(x)
        assert result.shape == (1, 4, 8)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64)
        x = Tensor(np.random.randn(2, 10, 32).astype(np.float32))
        result = block(x)
        assert result.shape == (2, 10, 32)

    def test_is_module(self):
        assert isinstance(TransformerBlock(16, 4, 32), Module)

    def test_has_parameters(self):
        block = TransformerBlock(embed_dim=16, num_heads=4, ff_dim=32)
        params = block.parameters()
        assert len(params) > 0

    def test_backward(self):
        block = TransformerBlock(embed_dim=8, num_heads=2, ff_dim=16)
        x = Tensor(np.random.randn(1, 3, 8).astype(np.float32), requires_grad=True)
        result = block(x)
        result.sum().backward()
        assert x.grad is not None


class TestTransformer:
    def test_output_shape(self):
        model = Transformer(vocab_size=100, embed_dim=32, num_heads=4,
                           ff_dim=64, num_layers=2, max_seq_len=128)
        x = Tensor([[1, 2, 3, 4]])
        result = model(x)
        assert result.shape == (1, 4, 100)  # (batch, seq_len, vocab_size)

    def test_is_module(self):
        model = Transformer(100, 16, 4, 32, 2, 64)
        assert isinstance(model, Module)

    def test_backward(self):
        model = Transformer(vocab_size=50, embed_dim=8, num_heads=2,
                           ff_dim=16, num_layers=1, max_seq_len=32)
        x = Tensor([[1, 2, 3]])
        result = model(x)
        result.sum().backward()
        params = model.parameters()
        assert any(p.grad is not None for p in params)


# ── State Dict ─────────────────────────────────────────────────────


class TestModuleStateDict:
    """Tests for Module.state_dict() and load_state_dict()."""

    def test_state_dict_returns_tensor_data(self):
        """state_dict returns {name: np.ndarray} for direct Tensor/Parameter attrs."""
        class M(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(Tensor([1.0, 2.0, 3.0]))
                self.bias = Parameter(Tensor([0.5]))
            def forward(self, x): return x

        m = M()
        sd = m.state_dict()
        assert "weight" in sd
        assert "bias" in sd
        np.testing.assert_array_equal(sd["weight"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(sd["bias"], [0.5])

    def test_state_dict_returns_numpy_arrays(self):
        """state_dict values are raw numpy arrays, not Tensor objects."""
        class M(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter(Tensor([1.0]))
            def forward(self, x): return x

        m = M()
        sd = m.state_dict()
        assert isinstance(sd["w"], np.ndarray)

    def test_state_dict_includes_plain_tensors(self):
        """state_dict includes plain Tensors (e.g. running_mean in BatchNorm)."""
        class M(Module):
            def __init__(self):
                super().__init__()
                self.running_mean = Tensor([0.0, 0.0])
            def forward(self, x): return x

        m = M()
        sd = m.state_dict()
        assert "running_mean" in sd

    def test_state_dict_skips_non_tensor_attrs(self):
        """state_dict ignores int, float, bool, string attributes."""
        class M(Module):
            def __init__(self):
                super().__init__()
                self.dim = 10
                self.eps = 1e-5
                self.w = Parameter(Tensor([1.0]))
            def forward(self, x): return x

        m = M()
        sd = m.state_dict()
        assert "dim" not in sd
        assert "eps" not in sd
        assert "w" in sd

    def test_state_dict_recurses_into_child_modules(self):
        """state_dict flattens child module params with dot notation."""
        class Child(Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter(Tensor([1.0, 2.0]))
            def forward(self, x): return x

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.child = Child()
                self.own = Parameter(Tensor([9.0]))
            def forward(self, x): return x

        m = Parent()
        sd = m.state_dict()
        assert "own" in sd
        assert "child.w" in sd
        np.testing.assert_array_equal(sd["child.w"], [1.0, 2.0])

    def test_state_dict_handles_list_of_modules(self):
        """state_dict recurses into lists of Modules with index in key."""
        class Child(Module):
            def __init__(self, val):
                super().__init__()
                self.w = Parameter(Tensor([val]))
            def forward(self, x): return x

        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.layers = [Child(1.0), Child(2.0)]
            def forward(self, x): return x

        m = Parent()
        sd = m.state_dict()
        assert "layers.0.w" in sd
        assert "layers.1.w" in sd
        np.testing.assert_array_equal(sd["layers.0.w"], [1.0])
        np.testing.assert_array_equal(sd["layers.1.w"], [2.0])

    def test_state_dict_handles_tuple_of_modules(self):
        """state_dict works with tuples (e.g. Sequential.modules)."""
        model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))
        sd = model.state_dict()
        assert "modules.0.weight" in sd
        assert "modules.0.bias" in sd
        assert "modules.2.weight" in sd
        assert "modules.2.bias" in sd
        # ReLU has no parameters, so no modules.1.* keys
        assert not any(k.startswith("modules.1.") for k in sd)

    def test_state_dict_linear_layer(self):
        """Linear layer state_dict has weight and bias."""
        layer = Linear(3, 5)
        sd = layer.state_dict()
        assert "weight" in sd
        assert "bias" in sd
        assert sd["weight"].shape == (3, 5)
        assert sd["bias"].shape == (1, 5)

    def test_state_dict_batchnorm_includes_running_stats(self):
        """BatchNorm state_dict includes running_mean and running_var."""
        bn = BatchNorm1d(4)
        sd = bn.state_dict()
        assert "weight" in sd
        assert "bias" in sd
        assert "running_mean" in sd
        assert "running_var" in sd

    def test_save_creates_npz_file(self, tmp_path):
        """save() writes an npz file to disk."""
        layer = Linear(3, 5)
        path = tmp_path / "model.npz"
        layer.save(str(path))
        assert (tmp_path / "model.npz").exists()

    def test_save_npz_contains_state_dict_keys(self, tmp_path):
        """Saved npz file contains the same keys as state_dict."""
        layer = Linear(3, 5)
        path = tmp_path / "model.npz"
        layer.save(str(path))
        loaded = np.load(str(path))
        sd = layer.state_dict()
        assert set(loaded.files) == set(sd.keys())
