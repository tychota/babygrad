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
        y = np.random.randint(0, num_classes, size=(batch_size,))
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


class TestTrainerFit:
    def test_fit_reduces_loss(self):
        """After a few epochs, loss should decrease."""
        from babygrad.trainer import Trainer

        np.random.seed(42)
        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.1)
        loss_fn = SoftmaxLoss()
        train_loader = make_fake_loader(num_batches=5, batch_size=8)

        trainer = Trainer(model, optimizer, loss_fn, train_loader)

        # Compute initial loss
        x, y = train_loader[0]
        initial_loss = loss_fn(model(x), y).data

        trainer.fit(5)

        # Compute final loss on same batch
        final_loss = loss_fn(model(x), y).data
        assert final_loss < initial_loss

    def test_fit_sets_model_to_train_mode(self):
        """fit() should call model.train() each epoch."""
        from babygrad.trainer import Trainer

        model = SimpleModel(2, 3)
        model.eval()  # Start in eval mode
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()
        train_loader = make_fake_loader(num_batches=1)

        trainer = Trainer(model, optimizer, loss_fn, train_loader)
        trainer.fit(1)

        assert model.training is True

    def test_fit_prints_epoch_summary(self, capsys):
        """fit() should print avg loss per epoch."""
        from babygrad.trainer import Trainer

        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()
        train_loader = make_fake_loader(num_batches=2)

        trainer = Trainer(model, optimizer, loss_fn, train_loader)
        trainer.fit(2)

        captured = capsys.readouterr().out
        assert "Epoch 1/2" in captured
        assert "Epoch 2/2" in captured
        assert "Avg Loss:" in captured
