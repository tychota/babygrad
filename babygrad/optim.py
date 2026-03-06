import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data = p.data - self.lr * p.grad


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                grad = param.grad
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param.data
                mt = self.m.get(param, 0) * self.beta1 + (1 - self.beta1) * grad
                self.m[param] = mt
                vt = self.v.get(param, 0) * self.beta2 + (1 - self.beta2) * (grad ** 2)
                self.v[param] = vt
                mt_hat = mt / (1 - self.beta1 ** self.t)
                vt_hat = vt / (1 - self.beta2 ** self.t)
                param.data -= self.lr * mt_hat / (vt_hat ** 0.5 + self.eps)


def clip_grad_norm(params, max_norm):
    """Clip gradient norm across all parameters. Returns the total norm."""
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += np.sum(p.grad ** 2)
    total_norm = float(np.sqrt(total_norm_sq))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total_norm
