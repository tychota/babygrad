import numpy as np
import pytest

from babygrad.tensor import Tensor
from babygrad.nn import Module, Linear, Sequential, Flatten, Parameter, SoftmaxLoss
from babygrad.optim import SGD


class SimpleModel(Module):
    """A tiny model for testing: single linear layer."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def make_fake_loader(num_batches=3, batch_size=4, in_features=2, num_classes=3):
    """Returns a list of (x, y) tuples acting as a DataLoader."""
    batches = []
    for _ in range(num_batches):
        x = Tensor(np.random.randn(batch_size, in_features).astype(np.float32))
        y = Tensor(np.random.randint(0, num_classes, size=(batch_size,)).astype(np.float32))
        batches.append((x, y))
    return batches


class TestTrainerInit:
    def test_stores_all_attributes(self):
        from babygrad.trainer import Trainer

        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()
        train_loader = make_fake_loader()
        val_loader = make_fake_loader()

        trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader)

        assert trainer.model is model
        assert trainer.optimizer is optimizer
        assert trainer.loss_fn is loss_fn
        assert trainer.train_loader is train_loader
        assert trainer.val_loader is val_loader

    def test_val_loader_defaults_to_none(self):
        from babygrad.trainer import Trainer

        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()
        train_loader = make_fake_loader()

        trainer = Trainer(model, optimizer, loss_fn, train_loader)

        assert trainer.val_loader is None
