import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError

    def state_dict(self):
        """Return optimizer state as a dict of numpy arrays."""
        return {}

    def load_state_dict(self, state_dict):
        """Load optimizer state from a dict."""
        pass

    def save(self, filename):
        """Save optimizer state to an npz file."""
        sd = self.state_dict()
        if sd:
            np.savez(filename, **{k: np.array(v) for k, v in sd.items()})

    def load(self, filename):
        """Load optimizer state from an npz file."""
        data = np.load(filename)
        loaded = {}
        for k in data.files:
            v = data[k]
            loaded[k] = int(v) if v.ndim == 0 else v
        self.load_state_dict(loaded)


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

    def state_dict(self):
        sd = {"t": self.t}
        for i, param in enumerate(self.params):
            if param in self.m:
                sd[f"m_{i}"] = self.m[param]
            if param in self.v:
                sd[f"v_{i}"] = self.v[param]
        return sd

    def load_state_dict(self, state_dict):
        self.t = state_dict["t"]
        for i, param in enumerate(self.params):
            if f"m_{i}" in state_dict:
                self.m[param] = state_dict[f"m_{i}"]
            if f"v_{i}" in state_dict:
                self.v[param] = state_dict[f"v_{i}"]


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


class CosineScheduler:
    """Cosine annealing with linear warmup."""
    def __init__(self, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / decay_steps if decay_steps > 0 else 1.0
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
