import numpy as np
import pytest

from babygrad.tensor import Tensor


def numerical_grad(f, x, eps=1e-5):
    """Compute numerical gradient via central differences."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        grad[idx] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


# ── Forward-only tests ──────────────────────────────────────────────


class TestAddForward:
    def test_add_two_tensors(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a + b
        np.testing.assert_array_almost_equal(c.data, [4.0, 6.0])

    def test_add_scalar_right(self):
        a = Tensor([1.0, 2.0])
        c = a + 10.0
        np.testing.assert_array_almost_equal(c.data, [11.0, 12.0])

    def test_add_scalar_left(self):
        a = Tensor([1.0, 2.0])
        c = 10.0 + a
        np.testing.assert_array_almost_equal(c.data, [11.0, 12.0])


class TestSubForward:
    def test_sub_two_tensors(self):
        a = Tensor([5.0, 7.0])
        b = Tensor([1.0, 2.0])
        c = a - b
        np.testing.assert_array_almost_equal(c.data, [4.0, 5.0])

    def test_sub_scalar_right(self):
        a = Tensor([5.0, 7.0])
        c = a - 1.0
        np.testing.assert_array_almost_equal(c.data, [4.0, 6.0])

    def test_sub_scalar_left(self):
        a = Tensor([5.0, 7.0])
        c = 10.0 - a
        np.testing.assert_array_almost_equal(c.data, [5.0, 3.0])


class TestMulForward:
    def test_mul_two_tensors(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = a * b
        np.testing.assert_array_almost_equal(c.data, [8.0, 15.0])

    def test_mul_scalar_right(self):
        a = Tensor([2.0, 3.0])
        c = a * 3.0
        np.testing.assert_array_almost_equal(c.data, [6.0, 9.0])

    def test_mul_scalar_left(self):
        a = Tensor([2.0, 3.0])
        c = 3.0 * a
        np.testing.assert_array_almost_equal(c.data, [6.0, 9.0])


class TestPowForward:
    def test_pow_integer_exponent(self):
        a = Tensor([2.0, 3.0])
        c = a ** 2
        np.testing.assert_array_almost_equal(c.data, [4.0, 9.0])

    def test_pow_fractional_exponent(self):
        a = Tensor([4.0, 9.0])
        c = a ** 0.5
        np.testing.assert_array_almost_equal(c.data, [2.0, 3.0])


class TestExpBaseForward:
    def test_rpow_base_2(self):
        a = Tensor([1.0, 2.0, 3.0])
        c = 2.0 ** a
        np.testing.assert_array_almost_equal(c.data, [2.0, 4.0, 8.0])

    def test_rpow_base_e(self):
        a = Tensor([0.0, 1.0])
        c = np.e ** a
        np.testing.assert_array_almost_equal(c.data, [1.0, np.e])


class TestTrueDivForward:
    def test_div_two_tensors(self):
        a = Tensor([6.0, 12.0])
        b = Tensor([2.0, 3.0])
        c = a / b
        np.testing.assert_array_almost_equal(c.data, [3.0, 4.0])

    def test_div_scalar_right(self):
        a = Tensor([6.0, 12.0])
        c = a / 3.0
        np.testing.assert_array_almost_equal(c.data, [2.0, 4.0])

    def test_div_scalar_left(self):
        a = Tensor([2.0, 4.0])
        c = 12.0 / a
        np.testing.assert_array_almost_equal(c.data, [6.0, 3.0])


class TestNegForward:
    def test_neg(self):
        a = Tensor([1.0, -2.0, 3.0])
        c = -a
        np.testing.assert_array_almost_equal(c.data, [-1.0, 2.0, -3.0])


class TestMatMulForward:
    def test_matmul_2d(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        expected = np.array([[1, 2], [3, 4]], dtype="float32") @ np.array(
            [[5, 6], [7, 8]], dtype="float32"
        )
        np.testing.assert_array_almost_equal(c.data, expected)


# ── Backward tests ──────────────────────────────────────────────────


class TestAddBackward:
    def test_add_gradient(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = a + b
        c.backward(Tensor([1.0, 1.0]))
        np.testing.assert_array_almost_equal(a.grad, [1.0, 1.0])
        np.testing.assert_array_almost_equal(b.grad, [1.0, 1.0])


class TestSubBackward:
    def test_sub_gradient(self):
        a = Tensor([5.0, 7.0], requires_grad=True)
        b = Tensor([1.0, 2.0], requires_grad=True)
        c = a - b
        c.backward(Tensor([1.0, 1.0]))
        np.testing.assert_array_almost_equal(a.grad, [1.0, 1.0])
        np.testing.assert_array_almost_equal(b.grad, [-1.0, -1.0])


class TestMulBackward:
    def test_mul_gradient(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        c.backward(Tensor([1.0, 1.0]))
        # d(a*b)/da = b, d(a*b)/db = a
        np.testing.assert_array_almost_equal(a.grad, [4.0, 5.0])
        np.testing.assert_array_almost_equal(b.grad, [2.0, 3.0])


class TestPowBackward:
    def test_pow_gradient(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        c = a ** 3
        c.backward(Tensor([1.0, 1.0]))
        # d(a^3)/da = 3*a^2
        np.testing.assert_array_almost_equal(a.grad, [12.0, 27.0])

    def test_pow_gradient_vs_numerical(self):
        x_np = np.array([2.0, 3.0], dtype="float64")

        def f(x):
            return np.sum(x ** 3)

        a = Tensor(x_np.copy(), dtype="float64", requires_grad=True)
        c = a ** 3
        c.backward(Tensor([1.0, 1.0], dtype="float64"))
        expected = numerical_grad(f, x_np)
        np.testing.assert_array_almost_equal(a.grad, expected, decimal=4)


class TestExpBaseBackward:
    def test_rpow_gradient(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        c = 2.0 ** a
        c.backward(Tensor([1.0, 1.0]))
        # d(2^a)/da = 2^a * ln(2)
        expected = (2.0 ** np.array([1.0, 2.0])) * np.log(2.0)
        np.testing.assert_array_almost_equal(a.grad, expected, decimal=5)


class TestTrueDivBackward:
    def test_div_gradient(self):
        a = Tensor([6.0, 12.0], requires_grad=True)
        b = Tensor([2.0, 3.0], requires_grad=True)
        c = a / b
        c.backward(Tensor([1.0, 1.0]))
        # d(a/b)/da = 1/b
        np.testing.assert_array_almost_equal(a.grad, [0.5, 1.0 / 3.0], decimal=5)
        # d(a/b)/db = -a/b^2
        np.testing.assert_array_almost_equal(b.grad, [-6.0 / 4.0, -12.0 / 9.0], decimal=5)


class TestNegBackward:
    def test_neg_gradient(self):
        a = Tensor([1.0, -2.0, 3.0], requires_grad=True)
        c = -a
        c.backward(Tensor([1.0, 1.0, 1.0]))
        np.testing.assert_array_almost_equal(a.grad, [-1.0, -1.0, -1.0])


class TestBackwardGraph:
    def test_backward_raises_if_no_grad(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        with pytest.raises(RuntimeError):
            a.backward()

    def test_chain_rule(self):
        """y = (a + b) * b  =>  dy/da = b, dy/db = a + 2b"""
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a + b      # c = 5
        y = c * b      # y = 15
        y.backward(Tensor([1.0]))
        np.testing.assert_array_almost_equal(a.grad, [3.0])      # dy/da = b
        np.testing.assert_array_almost_equal(b.grad, [2.0 + 2 * 3.0])  # dy/db = a + 2b = 8


# ── Scalar ops (MulScalar, DivScalar, PowerScalar) ──────────────────


class TestMulScalarForward:
    def test_mul_scalar_forward(self):
        from babygrad.ops import MulScalar
        a = Tensor([2.0, 3.0])
        c = MulScalar(5.0)(a)
        np.testing.assert_array_almost_equal(c.data, [10.0, 15.0])

    def test_mul_scalar_via_tensor(self):
        """tensor * float should use MulScalar."""
        a = Tensor([2.0, 3.0])
        c = a * 5.0
        np.testing.assert_array_almost_equal(c.data, [10.0, 15.0])


class TestMulScalarBackward:
    def test_mul_scalar_gradient(self):
        from babygrad.ops import MulScalar
        a = Tensor([2.0, 3.0], requires_grad=True)
        c = MulScalar(5.0)(a)
        c.backward(Tensor([1.0, 1.0]))
        # d(a*5)/da = 5
        np.testing.assert_array_almost_equal(a.grad, [5.0, 5.0])


class TestDivScalarForward:
    def test_div_scalar_forward(self):
        from babygrad.ops import DivScalar
        a = Tensor([10.0, 15.0])
        c = DivScalar(5.0)(a)
        np.testing.assert_array_almost_equal(c.data, [2.0, 3.0])

    def test_div_scalar_via_tensor(self):
        """tensor / float should use DivScalar."""
        a = Tensor([10.0, 15.0])
        c = a / 5.0
        np.testing.assert_array_almost_equal(c.data, [2.0, 3.0])


class TestDivScalarBackward:
    def test_div_scalar_gradient(self):
        from babygrad.ops import DivScalar
        a = Tensor([10.0, 15.0], requires_grad=True)
        c = DivScalar(5.0)(a)
        c.backward(Tensor([1.0, 1.0]))
        # d(a/5)/da = 1/5
        np.testing.assert_array_almost_equal(a.grad, [0.2, 0.2])


class TestPowerScalarForward:
    def test_power_scalar_forward(self):
        from babygrad.ops import PowerScalar
        a = Tensor([2.0, 3.0])
        c = PowerScalar(3.0)(a)
        np.testing.assert_array_almost_equal(c.data, [8.0, 27.0])

    def test_power_scalar_via_tensor(self):
        """tensor ** float should use PowerScalar."""
        a = Tensor([2.0, 3.0])
        c = a ** 3.0
        np.testing.assert_array_almost_equal(c.data, [8.0, 27.0])


class TestPowerScalarBackward:
    def test_power_scalar_gradient(self):
        from babygrad.ops import PowerScalar
        a = Tensor([2.0, 3.0], requires_grad=True)
        c = PowerScalar(3.0)(a)
        c.backward(Tensor([1.0, 1.0]))
        # d(a^3)/da = 3*a^2
        np.testing.assert_array_almost_equal(a.grad, [12.0, 27.0])

    def test_power_scalar_gradient_vs_numerical(self):
        x_np = np.array([2.0, 3.0], dtype="float64")
        a = Tensor(x_np.copy(), dtype="float64", requires_grad=True)
        c = a ** 3.0
        c.backward(Tensor([1.0, 1.0], dtype="float64"))

        expected = numerical_grad(lambda x: np.sum(x ** 3), x_np)
        np.testing.assert_array_almost_equal(a.grad, expected, decimal=4)


# ── Activation ops (ReLU, Tanh, Sigmoid) ─────────────────────────────


class TestReLUForward:
    def test_relu_positive_unchanged(self):
        from babygrad.ops import relu
        a = Tensor([1.0, 2.0, 3.0])
        c = relu(a)
        np.testing.assert_array_almost_equal(c.data, [1.0, 2.0, 3.0])

    def test_relu_negative_zeroed(self):
        from babygrad.ops import relu
        a = Tensor([-1.0, -2.0, -3.0])
        c = relu(a)
        np.testing.assert_array_almost_equal(c.data, [0.0, 0.0, 0.0])

    def test_relu_mixed(self):
        from babygrad.ops import relu
        a = Tensor([-1.0, 0.0, 2.0])
        c = relu(a)
        np.testing.assert_array_almost_equal(c.data, [0.0, 0.0, 2.0])


class TestReLUBackward:
    def test_relu_gradient(self):
        from babygrad.ops import relu
        a = Tensor([-1.0, 0.0, 2.0, 3.0], requires_grad=True)
        c = relu(a)
        c.backward(Tensor([1.0, 1.0, 1.0, 1.0]))
        # grad is 0 where input <= 0, 1 where input > 0
        np.testing.assert_array_almost_equal(a.grad, [0.0, 0.0, 1.0, 1.0])


class TestTanhForward:
    def test_tanh_values(self):
        from babygrad.ops import tanh
        a = Tensor([0.0, 1.0, -1.0])
        c = tanh(a)
        np.testing.assert_array_almost_equal(c.data, np.tanh([0.0, 1.0, -1.0]), decimal=5)


class TestTanhBackward:
    def test_tanh_gradient(self):
        from babygrad.ops import tanh
        a = Tensor([0.0, 1.0, -1.0], requires_grad=True)
        c = tanh(a)
        c.backward(Tensor([1.0, 1.0, 1.0]))
        # d(tanh)/da = 1 - tanh(a)^2
        expected = 1.0 - np.tanh([0.0, 1.0, -1.0]) ** 2
        np.testing.assert_array_almost_equal(a.grad, expected, decimal=5)


class TestSigmoidForward:
    def test_sigmoid_values(self):
        from babygrad.ops import sigmoid
        a = Tensor([0.0, 2.0, -2.0])
        c = sigmoid(a)
        expected = 1.0 / (1.0 + np.exp(-np.array([0.0, 2.0, -2.0])))
        np.testing.assert_array_almost_equal(c.data, expected, decimal=5)


class TestSigmoidBackward:
    def test_sigmoid_gradient(self):
        from babygrad.ops import sigmoid
        a = Tensor([0.0, 2.0, -2.0], requires_grad=True)
        c = sigmoid(a)
        c.backward(Tensor([1.0, 1.0, 1.0]))
        # d(sigmoid)/da = sigmoid(a) * (1 - sigmoid(a))
        s = 1.0 / (1.0 + np.exp(-np.array([0.0, 2.0, -2.0])))
        expected = s * (1.0 - s)
        np.testing.assert_array_almost_equal(a.grad, expected, decimal=5)


class TestGELUForward:
    def test_gelu_zero(self):
        from babygrad.ops import gelu
        a = Tensor([0.0])
        result = gelu(a)
        np.testing.assert_allclose(result.data, [0.0], atol=1e-5)
    def test_gelu_positive(self):
        from babygrad.ops import gelu
        a = Tensor([1.0, 2.0])
        result = gelu(a)
        x = np.array([1.0, 2.0])
        expected = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        np.testing.assert_allclose(result.data, expected, rtol=1e-5)
    def test_gelu_negative_small(self):
        from babygrad.ops import gelu
        a = Tensor([-1.0])
        result = gelu(a)
        assert result.data[0] < 0

class TestGELUBackward:
    def test_gelu_gradient(self):
        from babygrad.ops import gelu
        x_np = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        a = Tensor(x_np, requires_grad=True)
        result = gelu(a)
        result.sum().backward()
        assert a.grad is not None
        assert a.grad.shape == (3,)

class TestSiLUForward:
    def test_silu_zero(self):
        from babygrad.ops import silu
        a = Tensor([0.0])
        result = silu(a)
        np.testing.assert_allclose(result.data, [0.0], atol=1e-5)
    def test_silu_values(self):
        from babygrad.ops import silu
        a = Tensor([1.0, 2.0])
        result = silu(a)
        x = np.array([1.0, 2.0])
        expected = x / (1 + np.exp(-x))
        np.testing.assert_allclose(result.data, expected, rtol=1e-5)

class TestSiLUBackward:
    def test_silu_gradient(self):
        from babygrad.ops import silu
        x_np = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        a = Tensor(x_np, requires_grad=True)
        result = silu(a)
        result.sum().backward()
        assert a.grad is not None


class TestSqrtForward:
    def test_sqrt_values(self):
        from babygrad.ops import sqrt
        a = Tensor([1.0, 4.0, 9.0, 16.0])
        c = sqrt(a)
        np.testing.assert_array_almost_equal(c.data, [1.0, 2.0, 3.0, 4.0], decimal=5)

    def test_sqrt_small_values(self):
        from babygrad.ops import sqrt
        a = Tensor([0.01, 0.25])
        c = sqrt(a)
        np.testing.assert_array_almost_equal(c.data, [0.1, 0.5], decimal=5)


class TestSqrtBackward:
    def test_sqrt_gradient(self):
        from babygrad.ops import sqrt
        a = Tensor([1.0, 4.0, 9.0], requires_grad=True)
        c = sqrt(a)
        c.backward(Tensor([1.0, 1.0, 1.0]))
        # d(sqrt(a))/da = 1 / (2 * sqrt(a))
        expected = 1.0 / (2.0 * np.sqrt([1.0, 4.0, 9.0]))
        np.testing.assert_array_almost_equal(a.grad, expected, decimal=5)


class TestLogSumExpForward:
    def test_logsumexp_1d(self):
        from babygrad.ops import logsumexp
        a = Tensor([1.0, 2.0, 3.0])
        c = logsumexp(a, axes=(0,))
        expected = np.log(np.sum(np.exp([1.0, 2.0, 3.0])))
        np.testing.assert_array_almost_equal(c.data, expected, decimal=5)

    def test_logsumexp_2d_axis1(self):
        from babygrad.ops import logsumexp
        a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        a = Tensor(a_np)
        c = logsumexp(a, axes=(1,))
        expected = np.log(np.sum(np.exp(a_np), axis=1))
        np.testing.assert_array_almost_equal(c.data, expected, decimal=4)

    def test_logsumexp_max_trick_stability(self):
        """Should not overflow with large values thanks to max trick."""
        from babygrad.ops import logsumexp
        a = Tensor([1000.0, 1001.0, 1002.0])
        c = logsumexp(a, axes=(0,))
        expected = np.log(np.sum(np.exp(np.array([1000.0, 1001.0, 1002.0]) - 1002.0))) + 1002.0
        np.testing.assert_array_almost_equal(c.data, expected, decimal=3)


class TestLogSumExpBackward:
    def test_logsumexp_gradient(self):
        from babygrad.ops import logsumexp
        a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        c = logsumexp(a, axes=(1,))
        c.backward(Tensor([1.0]))
        # d(logsumexp)/dx_i = softmax(x)_i
        exp_a = np.exp(np.array([[1.0, 2.0, 3.0]]))
        expected = exp_a / np.sum(exp_a, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(a.grad, expected, decimal=5)

# ── Exp / Log tests ─────────────────────────────────────────────────


class TestExpForward:
    def test_exp_values(self):
        from babygrad.ops import exp
        a = Tensor([0.0, 1.0, 2.0])
        result = exp(a)
        np.testing.assert_allclose(result.data, np.exp([0.0, 1.0, 2.0]), rtol=1e-5)
    def test_exp_negative(self):
        from babygrad.ops import exp
        a = Tensor([-1.0, -2.0])
        result = exp(a)
        np.testing.assert_allclose(result.data, np.exp([-1.0, -2.0]), rtol=1e-5)

class TestExpBackward:
    def test_exp_gradient(self):
        from babygrad.ops import exp
        x_np = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        a = Tensor(x_np, dtype="float64", requires_grad=True)
        result = exp(a)
        result.sum().backward()
        expected = numerical_grad(lambda x: np.sum(np.exp(x)), x_np)
        np.testing.assert_allclose(a.grad, expected, rtol=1e-4)

class TestLogForward:
    def test_log_values(self):
        from babygrad.ops import log
        a = Tensor([1.0, 2.0, np.e])
        result = log(a)
        np.testing.assert_allclose(result.data, np.log([1.0, 2.0, np.e]), rtol=1e-5)

class TestLogBackward:
    def test_log_gradient(self):
        from babygrad.ops import log
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a = Tensor(x_np, dtype="float64", requires_grad=True)
        result = log(a)
        result.sum().backward()
        expected = numerical_grad(lambda x: np.sum(np.log(x)), x_np)
        np.testing.assert_allclose(a.grad, expected, rtol=1e-4)


# ── Max tests ────────────────────────────────────────────────────────


class TestMaxForward:
    def test_max_1d(self):
        from babygrad.ops import max
        a = Tensor([1.0, 3.0, 2.0])
        result = max(a)
        np.testing.assert_allclose(result.data, 3.0)
    def test_max_2d_axis1(self):
        from babygrad.ops import max
        a = Tensor([[1.0, 3.0], [4.0, 2.0]])
        result = max(a, axis=1)
        np.testing.assert_allclose(result.data, [3.0, 4.0])
    def test_max_keepdims(self):
        from babygrad.ops import max
        a = Tensor([[1.0, 3.0], [4.0, 2.0]])
        result = max(a, axis=1, keepdims=True)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.data, [[3.0], [4.0]])

class TestMaxBackward:
    def test_max_gradient(self):
        from babygrad.ops import max
        x_np = np.array([[1.0, 3.0], [4.0, 2.0]], dtype=np.float32)
        a = Tensor(x_np, requires_grad=True)
        result = max(a, axis=1)
        result.sum().backward()
        np.testing.assert_allclose(a.grad, [[0, 1], [1, 0]])


# ── Softmax tests ───────────────────────────────────────────────────


class TestSoftmaxForward:
    def test_softmax_sums_to_one(self):
        from babygrad.ops import softmax
        a = Tensor([[1.0, 2.0, 3.0]])
        result = softmax(a, axis=1)
        np.testing.assert_allclose(np.sum(result.data, axis=1), [1.0], rtol=1e-5)
    def test_softmax_values(self):
        from babygrad.ops import softmax
        a = Tensor([[1.0, 2.0, 3.0]])
        result = softmax(a, axis=1)
        expected = np.exp([1, 2, 3]) / np.sum(np.exp([1, 2, 3]))
        np.testing.assert_allclose(result.data[0], expected, rtol=1e-5)
    def test_softmax_numerical_stability(self):
        from babygrad.ops import softmax
        a = Tensor([[1000.0, 1001.0, 1002.0]])
        result = softmax(a, axis=1)
        np.testing.assert_allclose(np.sum(result.data, axis=1), [1.0], rtol=1e-5)

class TestSoftmaxBackward:
    def test_softmax_gradient(self):
        from babygrad.ops import softmax
        x_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        a = Tensor(x_np, requires_grad=True)
        result = softmax(a, axis=1)
        loss = (result * Tensor([[1.0, 0.0, 0.0]])).sum()
        loss.backward()
        assert a.grad is not None
        assert a.grad.shape == (1, 3)


# ── Where / Tril tests ──────────────────────────────────────────────


class TestWhereForward:
    def test_where_basic(self):
        from babygrad.ops import where
        cond = Tensor([1.0, 0.0, 1.0])
        a = Tensor([10.0, 20.0, 30.0])
        b = Tensor([1.0, 2.0, 3.0])
        result = where(cond, a, b)
        np.testing.assert_array_equal(result.data, [10.0, 2.0, 30.0])
    def test_where_2d(self):
        from babygrad.ops import where
        cond = Tensor([[1.0, 0.0], [0.0, 1.0]])
        a = Tensor([[10.0, 20.0], [30.0, 40.0]])
        b = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = where(cond, a, b)
        np.testing.assert_array_equal(result.data, [[10.0, 2.0], [3.0, 40.0]])

class TestWhereBackward:
    def test_where_gradient(self):
        from babygrad.ops import where
        cond = Tensor([1.0, 0.0, 1.0])
        a = Tensor([10.0, 20.0, 30.0], requires_grad=True)
        b = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = where(cond, a, b)
        result.sum().backward()
        np.testing.assert_array_equal(a.grad, [1.0, 0.0, 1.0])
        np.testing.assert_array_equal(b.grad, [0.0, 1.0, 0.0])

class TestTrilForward:
    def test_tril_basic(self):
        from babygrad.ops import tril
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = tril(a)
        expected = np.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    def test_tril_with_k(self):
        from babygrad.ops import tril
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = tril(a, k=1)
        expected = np.tril(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), k=1)
        np.testing.assert_array_equal(result.data, expected)
    def test_tril_batched(self):
        from babygrad.ops import tril
        a = Tensor(np.ones((2, 3, 3)))
        result = tril(a)
        expected = np.tril(np.ones((2, 3, 3)))
        np.testing.assert_array_equal(result.data, expected)

class TestTrilBackward:
    def test_tril_gradient(self):
        from babygrad.ops import tril
        x_np = np.ones((3, 3), dtype=np.float32)
        a = Tensor(x_np, requires_grad=True)
        result = tril(a)
        result.sum().backward()
        expected = np.tril(np.ones((3, 3)))
        np.testing.assert_array_equal(a.grad, expected)


# ── Concat / Split / Gather tests ───────────────────────────────────


class TestConcatForward:
    def test_concat_axis0(self):
        from babygrad.ops import concat
        a = Tensor([[1.0, 2.0]])
        b = Tensor([[3.0, 4.0]])
        result = concat([a, b], axis=0)
        np.testing.assert_array_equal(result.data, [[1, 2], [3, 4]])
    def test_concat_axis1(self):
        from babygrad.ops import concat
        a = Tensor([[1.0], [2.0]])
        b = Tensor([[3.0], [4.0]])
        result = concat([a, b], axis=1)
        np.testing.assert_array_equal(result.data, [[1, 3], [2, 4]])

class TestConcatBackward:
    def test_concat_gradient(self):
        from babygrad.ops import concat
        a = Tensor([[1.0, 2.0]], requires_grad=True)
        b = Tensor([[3.0, 4.0]], requires_grad=True)
        result = concat([a, b], axis=0)
        result.sum().backward()
        np.testing.assert_array_equal(a.grad, [[1, 1]])
        np.testing.assert_array_equal(b.grad, [[1, 1]])

class TestSplitForward:
    def test_split_equal(self):
        from babygrad.ops import split
        a = Tensor([[1.0, 2.0, 3.0, 4.0]])
        parts = split(a, 2, axis=1)
        assert len(parts) == 2
        np.testing.assert_array_equal(parts[0].data, [[1, 2]])
        np.testing.assert_array_equal(parts[1].data, [[3, 4]])

class TestSplitBackward:
    def test_split_gradient(self):
        from babygrad.ops import split
        a = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        parts = split(a, 2, axis=1)
        loss = parts[0].sum() + parts[1].sum()
        loss.backward()
        np.testing.assert_array_equal(a.grad, [[1, 1, 1, 1]])

class TestGatherForward:
    def test_gather_1d(self):
        from babygrad.ops import gather
        a = Tensor([10.0, 20.0, 30.0, 40.0])
        indices = np.array([0, 2, 3])
        result = gather(a, indices, axis=0)
        np.testing.assert_array_equal(result.data, [10, 30, 40])
    def test_gather_2d_axis0(self):
        from babygrad.ops import gather
        a = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        indices = np.array([0, 2])
        result = gather(a, indices, axis=0)
        np.testing.assert_array_equal(result.data, [[1, 2], [5, 6]])

class TestGatherBackward:
    def test_gather_gradient(self):
        from babygrad.ops import gather
        a = Tensor([10.0, 20.0, 30.0, 40.0], requires_grad=True)
        indices = np.array([0, 2, 0])
        result = gather(a, indices, axis=0)
        result.sum().backward()
        np.testing.assert_array_equal(a.grad, [2, 0, 1, 0])


# ── Embedding op tests ──────────────────────────────────────────────


class TestEmbeddingOpForward:
    def test_embedding_lookup(self):
        from babygrad.ops import embedding
        weight = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        indices = Tensor([0, 2, 1])
        result = embedding(weight, indices)
        expected = np.array([[0.1, 0.2], [0.5, 0.6], [0.3, 0.4]])
        np.testing.assert_allclose(result.data, expected, rtol=1e-5)
    def test_embedding_2d_indices(self):
        from babygrad.ops import embedding
        weight = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        indices = Tensor([[0, 1], [2, 0]])
        result = embedding(weight, indices)
        assert result.shape == (2, 2, 2)

class TestEmbeddingOpBackward:
    def test_embedding_gradient(self):
        from babygrad.ops import embedding
        weight = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
        indices = Tensor([0, 2, 0])
        result = embedding(weight, indices)
        result.sum().backward()
        expected = np.array([[2, 2], [0, 0], [1, 1]], dtype=np.float32)
        np.testing.assert_allclose(weight.grad, expected)
