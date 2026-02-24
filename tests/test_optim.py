import pytest
import numpy as np
from babygrad import Tensor
from babygrad.optim import Optimizer, SGD


# ── Optimizer base class ──


def test_optimizer_stores_params():
    p1 = Tensor([1.0, 2.0], requires_grad=True)
    p2 = Tensor([3.0], requires_grad=True)
    opt = Optimizer([p1, p2])
    assert opt.params == [p1, p2]


def test_zero_grad_clears_gradients():
    p1 = Tensor([1.0, 2.0], requires_grad=True)
    p2 = Tensor([3.0], requires_grad=True)
    p1.grad = np.array([0.1, 0.2])
    p2.grad = np.array([0.3])
    opt = Optimizer([p1, p2])
    opt.zero_grad()
    assert p1.grad is None
    assert p2.grad is None


def test_zero_grad_when_already_none():
    p = Tensor([1.0], requires_grad=True)
    assert p.grad is None
    opt = Optimizer([p])
    opt.zero_grad()  # should not raise
    assert p.grad is None


def test_step_raises_not_implemented():
    p = Tensor([1.0], requires_grad=True)
    opt = Optimizer([p])
    with pytest.raises(NotImplementedError):
        opt.step()


# ── SGD ──


def test_sgd_step_updates_params():
    p = Tensor([4.0, 6.0], requires_grad=True)
    p.grad = np.array([2.0, 3.0])
    opt = SGD([p], lr=0.1)
    opt.step()
    np.testing.assert_allclose(p.data, [3.8, 5.7])


def test_sgd_default_lr():
    opt = SGD([Tensor([1.0], requires_grad=True)])
    assert opt.lr == 0.01


def test_sgd_custom_lr():
    opt = SGD([Tensor([1.0], requires_grad=True)], lr=0.5)
    assert opt.lr == 0.5


def test_sgd_skips_none_grad():
    p = Tensor([1.0, 2.0], requires_grad=True)
    assert p.grad is None
    opt = SGD([p], lr=0.1)
    opt.step()  # should not raise
    np.testing.assert_allclose(p.data, [1.0, 2.0])


def test_sgd_is_optimizer():
    opt = SGD([Tensor([1.0], requires_grad=True)])
    assert isinstance(opt, Optimizer)


def test_sgd_full_loop():
    """Integration: zero_grad → backward → step."""
    w = Tensor([2.0], requires_grad=True)
    opt = SGD([w], lr=0.1)

    # simulate: loss = w * 3  →  dloss/dw = 3
    w.grad = np.array([3.0])

    opt.step()
    np.testing.assert_allclose(w.data, [1.7])  # 2.0 - 0.1*3.0

    opt.zero_grad()
    assert w.grad is None
