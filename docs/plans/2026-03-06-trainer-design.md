# Trainer Design

## Overview
A `Trainer` class that wraps the training loop boilerplate.

## API

### `Trainer.__init__(model, optimizer, loss_fn, train_loader, val_loader=None)`
Stores all five attributes. No weight initialization — caller handles that.

### `Trainer.fit(epochs: int)`
For each epoch:
1. `model.train()`
2. Loop over `train_loader`, unpack `(x, y)` tuples
3. `optimizer.zero_grad()` → `model(x)` → `loss_fn(pred, y)` → `loss.backward()` → `optimizer.step()`
4. Track `total_loss` and `num_batches`
5. Print `Epoch {n}/{epochs} - Avg Loss: {avg:.4f}` + val accuracy if `val_loader` exists

### `Trainer.evaluate(loader=None) -> float`
Uses `val_loader` by default. Returns 0.0 if no loader available.
1. `model.eval()`
2. Loop over loader, forward pass, `argmax(axis=1)` on `.data`
3. Return `correct / total`

## Constraints
- Tuples only — no `batch.x`/`batch.y` attribute access
- No auto-Tensor wrapping (DataLoader already does it)
- No batch-level logging — epoch-level only
- No weight initialization — external responsibility
- Minimal printing: one line per epoch

## Integration
- New file: `babygrad/trainer.py`
- Add `Trainer` to `babygrad/__init__.py` imports and `__all__`
