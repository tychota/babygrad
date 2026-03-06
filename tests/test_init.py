import math
import numpy as np
import pytest

from babygrad.tensor import Tensor
from babygrad.init import xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, gain_for_nonlinearity


class TestGainForNonlinearity:
    """Tests for gain_for_nonlinearity helper."""

    def test_relu(self):
        assert gain_for_nonlinearity("relu") == math.sqrt(2.0)

    def test_tanh(self):
        assert gain_for_nonlinearity("tanh") == 5.0 / 3.0

    def test_sigmoid(self):
        assert gain_for_nonlinearity("sigmoid") == 1.0

    def test_gelu(self):
        assert gain_for_nonlinearity("gelu") == math.sqrt(2.0)

    def test_silu(self):
        assert gain_for_nonlinearity("silu") == math.sqrt(2.0)

    def test_linear(self):
        assert gain_for_nonlinearity("linear") == 1.0

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            gain_for_nonlinearity("unknown_activation")


class TestXavierUniform:
    """Tests for xavier_uniform initialization."""

    def test_returns_tensor(self):
        """xavier_uniform returns a Tensor."""
        t = xavier_uniform(fan_in=784, fan_out=256)
        assert isinstance(t, Tensor)

    def test_shape(self):
        """Output tensor has shape (fan_in, fan_out)."""
        t = xavier_uniform(fan_in=784, fan_out=256)
        assert t.shape == (784, 256)

    def test_bound_default_gain(self):
        """Values lie within [-a, a] where a = sqrt(6 / (fan_in + fan_out))."""
        fan_in, fan_out = 784, 256
        a = math.sqrt(6.0 / (fan_in + fan_out))
        t = xavier_uniform(fan_in=fan_in, fan_out=fan_out)
        assert np.all(t.data >= -a)
        assert np.all(t.data <= a)

    def test_bound_with_gain(self):
        """Values lie within [-gain*a, gain*a]."""
        fan_in, fan_out, gain = 784, 256, 2.0
        a = gain * math.sqrt(6.0 / (fan_in + fan_out))
        t = xavier_uniform(fan_in=fan_in, fan_out=fan_out, gain=gain)
        assert np.all(t.data >= -a)
        assert np.all(t.data <= a)

    def test_mean_near_zero(self):
        """Mean of a large tensor should be close to zero."""
        t = xavier_uniform(fan_in=1000, fan_out=1000)
        assert abs(float(np.mean(t.data))) < 0.01

    def test_requires_grad(self):
        """Returned tensor should require grad."""
        t = xavier_uniform(fan_in=10, fan_out=10)
        assert t.requires_grad is True

    def test_custom_shape(self):
        """shape parameter overrides default (fan_in, fan_out) shape."""
        t = xavier_uniform(fan_in=784, fan_out=256, shape=(3, 3, 64))
        assert t.shape == (3, 3, 64)



class TestXavierNormal:
    """Tests for xavier_normal initialization."""

    def test_returns_tensor(self):
        """xavier_normal returns a Tensor."""
        t = xavier_normal(fan_in=784, fan_out=256)
        assert isinstance(t, Tensor)

    def test_shape(self):
        """Output tensor has shape (fan_in, fan_out)."""
        t = xavier_normal(fan_in=784, fan_out=256)
        assert t.shape == (784, 256)

    def test_std_default_gain(self):
        """Std should be close to gain * sqrt(2 / (fan_in + fan_out))."""
        fan_in, fan_out = 1000, 1000
        expected_std = math.sqrt(2.0 / (fan_in + fan_out))
        t = xavier_normal(fan_in=fan_in, fan_out=fan_out)
        actual_std = float(np.std(t.data))
        assert abs(actual_std - expected_std) < 0.005

    def test_std_with_gain(self):
        """Std scales with gain."""
        fan_in, fan_out, gain = 1000, 1000, 2.0
        expected_std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        t = xavier_normal(fan_in=fan_in, fan_out=fan_out, gain=gain)
        actual_std = float(np.std(t.data))
        assert abs(actual_std - expected_std) < 0.01

    def test_mean_near_zero(self):
        """Mean should be close to zero."""
        t = xavier_normal(fan_in=1000, fan_out=1000)
        assert abs(float(np.mean(t.data))) < 0.01

    def test_requires_grad(self):
        """Returned tensor should require grad."""
        t = xavier_normal(fan_in=10, fan_out=10)
        assert t.requires_grad is True

    def test_custom_shape(self):
        """shape parameter overrides default (fan_in, fan_out) shape."""
        t = xavier_normal(fan_in=784, fan_out=256, shape=(3, 3, 64))
        assert t.shape == (3, 3, 64)



class TestKaimingUniform:
    """Tests for kaiming_uniform initialization."""

    def test_returns_tensor(self):
        """kaiming_uniform returns a Tensor."""
        t = kaiming_uniform(fan_in=784, fan_out=256)
        assert isinstance(t, Tensor)

    def test_shape(self):
        """Output tensor has shape (fan_in, fan_out)."""
        t = kaiming_uniform(fan_in=784, fan_out=256)
        assert t.shape == (784, 256)

    def test_bound_relu(self):
        """Values lie within [-a, a] where a = gain * sqrt(3 / fan_in) for relu."""
        fan_in, fan_out = 784, 256
        gain = math.sqrt(2.0)
        a = gain * math.sqrt(3.0 / fan_in)
        t = kaiming_uniform(fan_in=fan_in, fan_out=fan_out, nonlinearity="relu")
        assert np.all(t.data >= -a)
        assert np.all(t.data <= a)

    def test_bound_tanh(self):
        """Tanh gain (5/3) produces wider bounds than sigmoid (1.0)."""
        fan_in, fan_out = 784, 256
        gain_tanh = 5.0 / 3.0
        a_tanh = gain_tanh * math.sqrt(3.0 / fan_in)
        t = kaiming_uniform(fan_in=fan_in, fan_out=fan_out, nonlinearity="tanh")
        assert np.all(t.data >= -a_tanh)
        assert np.all(t.data <= a_tanh)

    def test_mean_near_zero(self):
        """Mean of a large tensor should be close to zero."""
        t = kaiming_uniform(fan_in=1000, fan_out=1000)
        assert abs(float(np.mean(t.data))) < 0.01

    def test_requires_grad(self):
        """Returned tensor should require grad."""
        t = kaiming_uniform(fan_in=10, fan_out=10)
        assert t.requires_grad is True

    def test_custom_shape(self):
        """shape parameter overrides default (fan_in, fan_out) shape."""
        t = kaiming_uniform(fan_in=784, fan_out=256, shape=(3, 3, 64))
        assert t.shape == (3, 3, 64)


class TestKaimingNormal:
    """Tests for kaiming_normal initialization."""

    def test_returns_tensor(self):
        """kaiming_normal returns a Tensor."""
        t = kaiming_normal(fan_in=784, fan_out=256)
        assert isinstance(t, Tensor)

    def test_shape(self):
        """Output tensor has shape (fan_in, fan_out)."""
        t = kaiming_normal(fan_in=784, fan_out=256)
        assert t.shape == (784, 256)

    def test_std_relu(self):
        """Std for relu should be close to sqrt(2) / sqrt(fan_in)."""
        fan_in, fan_out = 1000, 1000
        gain = math.sqrt(2.0)
        expected_std = gain / math.sqrt(fan_in)
        t = kaiming_normal(fan_in=fan_in, fan_out=fan_out, nonlinearity="relu")
        actual_std = float(np.std(t.data))
        assert abs(actual_std - expected_std) < 0.005

    def test_std_tanh(self):
        """Std for tanh should use gain=5/3."""
        fan_in, fan_out = 1000, 1000
        gain = 5.0 / 3.0
        expected_std = gain / math.sqrt(fan_in)
        t = kaiming_normal(fan_in=fan_in, fan_out=fan_out, nonlinearity="tanh")
        actual_std = float(np.std(t.data))
        assert abs(actual_std - expected_std) < 0.005

    def test_mean_near_zero(self):
        """Mean should be close to zero."""
        t = kaiming_normal(fan_in=1000, fan_out=1000)
        assert abs(float(np.mean(t.data))) < 0.01

    def test_requires_grad(self):
        """Returned tensor should require grad."""
        t = kaiming_normal(fan_in=10, fan_out=10)
        assert t.requires_grad is True

    def test_custom_shape(self):
        """shape parameter overrides default (fan_in, fan_out) shape."""
        t = kaiming_normal(fan_in=784, fan_out=256, shape=(3, 3, 64))
        assert t.shape == (3, 3, 64)

    def test_unknown_nonlinearity_raises(self):
        """Unknown nonlinearity raises ValueError."""
        with pytest.raises(ValueError):
            kaiming_normal(fan_in=10, fan_out=10, nonlinearity="unknown")
