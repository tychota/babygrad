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


class TestTrainerEvaluate:
    def test_evaluate_returns_zero_without_loader(self):
        """evaluate() returns 0.0 when no val_loader is set."""
        from babygrad.trainer import Trainer

        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()
        train_loader = make_fake_loader()

        trainer = Trainer(model, optimizer, loss_fn, train_loader)
        assert trainer.evaluate() == 0.0

    def test_evaluate_sets_model_to_eval_mode(self):
        """evaluate() should call model.eval()."""
        from babygrad.trainer import Trainer

        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()
        val_loader = make_fake_loader(num_batches=1)

        trainer = Trainer(model, optimizer, loss_fn, make_fake_loader(),
                          val_loader=val_loader)
        trainer.evaluate()

        assert model.training is False

    def test_evaluate_returns_accuracy(self):
        """evaluate() should return correct/total as a float."""
        from babygrad.trainer import Trainer

        # Build a model that always predicts class 0 via large bias
        model = SimpleModel(2, 3)
        model.linear.weight.data = np.zeros((2, 3), dtype=np.float32)
        model.linear.bias.data = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)

        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()

        # All labels are 0 -> 100% accuracy
        all_zero_loader = [
            (Tensor(np.random.randn(4, 2).astype(np.float32)),
             np.zeros(4, dtype=np.int64))
            for _ in range(2)
        ]

        trainer = Trainer(model, optimizer, loss_fn, make_fake_loader(),
                          val_loader=all_zero_loader)
        acc = trainer.evaluate()
        assert acc == 1.0

    def test_evaluate_with_explicit_loader(self):
        """evaluate(loader) should use the provided loader, not val_loader."""
        from babygrad.trainer import Trainer

        model = SimpleModel(2, 3)
        model.linear.weight.data = np.zeros((2, 3), dtype=np.float32)
        model.linear.bias.data = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)

        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()

        explicit_loader = [
            (Tensor(np.random.randn(4, 2).astype(np.float32)),
             np.zeros(4, dtype=np.int64))
        ]

        trainer = Trainer(model, optimizer, loss_fn, make_fake_loader())
        acc = trainer.evaluate(loader=explicit_loader)
        assert acc == 1.0

    def test_fit_prints_val_accuracy_when_val_loader_set(self, capsys):
        """fit() should print val accuracy when val_loader is provided."""
        from babygrad.trainer import Trainer

        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()
        train_loader = make_fake_loader(num_batches=1)
        val_loader = make_fake_loader(num_batches=1)

        trainer = Trainer(model, optimizer, loss_fn, train_loader,
                          val_loader=val_loader)
        trainer.fit(1)

        captured = capsys.readouterr().out
        assert "Val Acc:" in captured


class TestTrainerComputeMetrics:
    def test_compute_metrics_stored(self):
        """Trainer should store compute_metrics callback."""
        from babygrad.trainer import Trainer

        def my_metrics(model, loader):
            return 0.5

        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()

        trainer = Trainer(model, optimizer, loss_fn, make_fake_loader(),
                          compute_metrics=my_metrics)
        assert trainer.compute_metrics is my_metrics

    def test_compute_metrics_defaults_to_none(self):
        """compute_metrics should default to None."""
        from babygrad.trainer import Trainer

        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()

        trainer = Trainer(model, optimizer, loss_fn, make_fake_loader())
        assert trainer.compute_metrics is None

    def test_evaluate_uses_compute_metrics_when_provided(self):
        """evaluate() should call compute_metrics instead of default logic."""
        from babygrad.trainer import Trainer

        call_log = []

        def custom_metrics(model, loader):
            call_log.append((model, loader))
            return 0.42

        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()
        val_loader = make_fake_loader(num_batches=1)

        trainer = Trainer(model, optimizer, loss_fn, make_fake_loader(),
                          val_loader=val_loader, compute_metrics=custom_metrics)
        result = trainer.evaluate()

        assert result == 0.42
        assert len(call_log) == 1
        assert call_log[0][0] is model
        assert call_log[0][1] is val_loader

    def test_evaluate_passes_explicit_loader_to_compute_metrics(self):
        """evaluate(loader) should pass the explicit loader to compute_metrics."""
        from babygrad.trainer import Trainer

        received_loader = []

        def custom_metrics(model, loader):
            received_loader.append(loader)
            return 0.99

        model = SimpleModel(2, 3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()
        explicit_loader = make_fake_loader(num_batches=1)

        trainer = Trainer(model, optimizer, loss_fn, make_fake_loader(),
                          compute_metrics=custom_metrics)
        result = trainer.evaluate(loader=explicit_loader)

        assert result == 0.99
        assert received_loader[0] is explicit_loader

    def test_evaluate_still_sets_eval_mode_with_compute_metrics(self):
        """evaluate() should set eval mode even when using compute_metrics."""
        from babygrad.trainer import Trainer

        def custom_metrics(model, loader):
            return 0.5

        model = SimpleModel(2, 3)
        model.train()
        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = SoftmaxLoss()
        val_loader = make_fake_loader(num_batches=1)

        trainer = Trainer(model, optimizer, loss_fn, make_fake_loader(),
                          val_loader=val_loader, compute_metrics=custom_metrics)
        trainer.evaluate()

        assert model.training is False
