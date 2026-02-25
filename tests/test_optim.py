import pytest
import numpy as np
from babygrad import Tensor
from babygrad.optim import Optimizer, SGD, Adam


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


# ── Adam ──


def test_adam_is_optimizer():
    opt = Adam([Tensor([1.0], requires_grad=True)])
    assert isinstance(opt, Optimizer)


def test_adam_default_hyperparams():
    opt = Adam([Tensor([1.0], requires_grad=True)])
    assert opt.lr == 0.001
    assert opt.beta1 == 0.9
    assert opt.beta2 == 0.999
    assert opt.eps == 1e-8
    assert opt.weight_decay == 0.0


def test_adam_custom_hyperparams():
    opt = Adam(
        [Tensor([1.0], requires_grad=True)],
        lr=0.01, beta1=0.8, beta2=0.99, eps=1e-6, weight_decay=0.1,
    )
    assert opt.lr == 0.01
    assert opt.beta1 == 0.8
    assert opt.beta2 == 0.99
    assert opt.eps == 1e-6
    assert opt.weight_decay == 0.1


def test_adam_initializes_state():
    opt = Adam([Tensor([1.0], requires_grad=True)])
    assert opt.t == 0
    assert opt.m == {}
    assert opt.v == {}


def test_adam_skips_none_grad():
    p = Tensor([1.0, 2.0], requires_grad=True)
    assert p.grad is None
    opt = Adam([p])
    opt.step()  # should not raise
    np.testing.assert_allclose(p.data, [1.0, 2.0])


def test_adam_single_step():
    """Verify one Adam step matches hand-computed values."""
    p = Tensor([0.5], requires_grad=True)
    p.grad = np.array([0.1])
    opt = Adam([p], lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    opt.step()

    # t=1, grad=0.1
    # m1 = 0.9*0 + 0.1*0.1 = 0.01
    # v1 = 0.999*0 + 0.001*0.01 = 0.00001
    # m1_hat = 0.01 / (1 - 0.9) = 0.1
    # v1_hat = 0.00001 / (1 - 0.999) = 0.01
    # update = 0.001 * 0.1 / (sqrt(0.01) + 1e-8) = 0.001 * 0.1 / 0.1 = 0.001
    # p = 0.5 - 0.001 = 0.499
    np.testing.assert_allclose(p.data, [0.499], rtol=1e-6)
    assert opt.t == 1


def test_adam_two_steps():
    """Verify Adam accumulates moments across steps."""
    p = Tensor([1.0], requires_grad=True)
    opt = Adam([p], lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)

    # Step 1: grad=0.2
    p.grad = np.array([0.2])
    opt.step()

    # Step 2: grad=-0.1
    p.grad = np.array([-0.1])
    opt.step()

    # t=2
    # m2 = 0.9*(0.9*0+0.1*0.2) + 0.1*(-0.1) = 0.9*0.02 + (-0.01) = 0.008
    # v2 = 0.999*(0.999*0+0.001*0.04) + 0.001*0.01 = 0.999*0.00004 + 0.00001
    #    = 0.00003996 + 0.00001 = 0.00004996
    # m2_hat = 0.008 / (1 - 0.81) = 0.008 / 0.19 ≈ 0.04210526
    # v2_hat = 0.00004996 / (1 - 0.998001) = 0.00004996 / 0.001999 ≈ 0.02499250
    # update = 0.001 * 0.04210526 / (sqrt(0.02499250) + 1e-8)
    #        = 0.001 * 0.04210526 / 0.15809650 ≈ 0.000266326
    # After step1: p ≈ 0.999, after step2: p ≈ 0.999 - 0.000266326 ≈ 0.998734
    np.testing.assert_allclose(p.data, [0.998734], rtol=1e-4)
    assert opt.t == 2


def test_adam_weight_decay():
    """Weight decay adds param.data to the gradient."""
    p = Tensor([2.0], requires_grad=True)
    p.grad = np.array([0.1])
    opt = Adam([p], lr=0.001, weight_decay=0.1)
    opt.step()

    # effective grad = 0.1 + 0.1 * 2.0 = 0.3
    # m1 = 0.1 * 0.3 = 0.03
    # v1 = 0.001 * 0.09 = 0.00009
    # m1_hat = 0.03 / 0.1 = 0.3
    # v1_hat = 0.00009 / 0.001 = 0.09
    # update = 0.001 * 0.3 / (sqrt(0.09) + 1e-8) = 0.001 * 0.3 / 0.3 = 0.001
    np.testing.assert_allclose(p.data, [1.999], rtol=1e-6)


def test_adam_no_weight_decay_by_default():
    """Without weight_decay, grad is used as-is."""
    p_wd = Tensor([2.0], requires_grad=True)
    p_wd.grad = np.array([0.1])
    opt_wd = Adam([p_wd], lr=0.001, weight_decay=0.0)
    opt_wd.step()

    p_no = Tensor([2.0], requires_grad=True)
    p_no.grad = np.array([0.1])
    opt_no = Adam([p_no], lr=0.001)
    opt_no.step()

    np.testing.assert_allclose(p_wd.data, p_no.data)


def test_adam_multiple_params():
    """Adam tracks moments independently per parameter."""
    p1 = Tensor([1.0], requires_grad=True)
    p2 = Tensor([2.0], requires_grad=True)
    p1.grad = np.array([0.5])
    p2.grad = np.array([-0.3])
    opt = Adam([p1, p2], lr=0.001)
    opt.step()

    # Each param should have its own m and v entry
    assert len(opt.m) == 2
    assert len(opt.v) == 2
    # Both should have moved from their initial values
    assert p1.data[0] < 1.0  # positive grad → decrease
    assert p2.data[0] > 2.0  # negative grad → increase


def test_adam_full_loop():
    """Integration: zero_grad → set grad → step."""
    w = Tensor([3.0], requires_grad=True)
    opt = Adam([w], lr=0.001)

    w.grad = np.array([1.0])
    opt.step()
    after_step1 = w.data.copy()
    assert after_step1[0] < 3.0

    opt.zero_grad()
    assert w.grad is None

    w.grad = np.array([1.0])
    opt.step()
    assert w.data[0] < after_step1[0]
