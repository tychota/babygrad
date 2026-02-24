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
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a + b
        c.backward(Tensor([1.0, 1.0]))
        np.testing.assert_array_almost_equal(a.grad, [1.0, 1.0])
        np.testing.assert_array_almost_equal(b.grad, [1.0, 1.0])


class TestSubBackward:
    def test_sub_gradient(self):
        a = Tensor([5.0, 7.0])
        b = Tensor([1.0, 2.0])
        c = a - b
        c.backward(Tensor([1.0, 1.0]))
        np.testing.assert_array_almost_equal(a.grad, [1.0, 1.0])
        np.testing.assert_array_almost_equal(b.grad, [-1.0, -1.0])


class TestMulBackward:
    def test_mul_gradient(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = a * b
        c.backward(Tensor([1.0, 1.0]))
        # d(a*b)/da = b, d(a*b)/db = a
        np.testing.assert_array_almost_equal(a.grad, [4.0, 5.0])
        np.testing.assert_array_almost_equal(b.grad, [2.0, 3.0])


class TestPowBackward:
    def test_pow_gradient(self):
        a = Tensor([2.0, 3.0])
        c = a ** 3
        c.backward(Tensor([1.0, 1.0]))
        # d(a^3)/da = 3*a^2
        np.testing.assert_array_almost_equal(a.grad, [12.0, 27.0])

    def test_pow_gradient_vs_numerical(self):
        x_np = np.array([2.0, 3.0], dtype="float64")

        def f(x):
            return np.sum(x ** 3)

        a = Tensor(x_np.copy(), dtype="float64")
        c = a ** 3
        c.backward(Tensor([1.0, 1.0], dtype="float64"))
        expected = numerical_grad(f, x_np)
        np.testing.assert_array_almost_equal(a.grad, expected, decimal=4)


class TestExpBaseBackward:
    def test_rpow_gradient(self):
        a = Tensor([1.0, 2.0])
        c = 2.0 ** a
        c.backward(Tensor([1.0, 1.0]))
        # d(2^a)/da = 2^a * ln(2)
        expected = (2.0 ** np.array([1.0, 2.0])) * np.log(2.0)
        np.testing.assert_array_almost_equal(a.grad, expected, decimal=5)


class TestTrueDivBackward:
    def test_div_gradient(self):
        a = Tensor([6.0, 12.0])
        b = Tensor([2.0, 3.0])
        c = a / b
        c.backward(Tensor([1.0, 1.0]))
        # d(a/b)/da = 1/b
        np.testing.assert_array_almost_equal(a.grad, [0.5, 1.0 / 3.0], decimal=5)
        # d(a/b)/db = -a/b^2
        np.testing.assert_array_almost_equal(b.grad, [-6.0 / 4.0, -12.0 / 9.0], decimal=5)


class TestNegBackward:
    def test_neg_gradient(self):
        a = Tensor([1.0, -2.0, 3.0])
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
        a = Tensor([2.0])
        b = Tensor([3.0])
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
        a = Tensor([2.0, 3.0])
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
        a = Tensor([10.0, 15.0])
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
        a = Tensor([2.0, 3.0])
        c = PowerScalar(3.0)(a)
        c.backward(Tensor([1.0, 1.0]))
        # d(a^3)/da = 3*a^2
        np.testing.assert_array_almost_equal(a.grad, [12.0, 27.0])

    def test_power_scalar_gradient_vs_numerical(self):
        x_np = np.array([2.0, 3.0], dtype="float64")
        a = Tensor(x_np.copy(), dtype="float64")
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
