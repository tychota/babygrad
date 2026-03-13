# Modular Addition Example Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A readable example that trains a 1-layer Transformer to learn (A + B) % 113 and demonstrates grokking.

**Architecture:** Custom Dataset yields `([A, B, =], [-1, -1, C])` pairs. DataLoader batches them. Manual training loop with CrossEntropyLoss(ignore_index=-1). Evaluation checks argmax of last position.

**Tech Stack:** babygrad (Tensor, Transformer, CrossEntropyLoss, Adam, Dataset, DataLoader)

---

### Task 1: ModularAdditionDataset

**Files:**
- Create: `examples/modular_addition.py`

**Step 1: Write the dataset class and a quick smoke test at the bottom**

```python
"""
Modular Addition with a Transformer
====================================
Train a single-layer Transformer to learn (A + B) % 113.

This demonstrates the "grokking" phenomenon: the model memorizes the
training set quickly (train accuracy → 100%) but takes much longer to
generalize (test accuracy → 100%).

Vocabulary: 0-112 = numbers, 113 = "=" token
Input:  [A, B, =]   (3 tokens)
Target: [-1, -1, C]  (loss only on answer position, C = (A+B) % 113)
"""

import numpy as np
from babygrad import (
    Tensor, Transformer, CrossEntropyLoss, Adam, Dataset, DataLoader,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

P = 113          # prime modulus
EQ_TOKEN = P     # "=" token has id 113
VOCAB_SIZE = P + 1  # 0..112 + "="
SEQ_LEN = 3     # [A, B, =]
IGNORE = -1      # ignore index for loss

class ModularAdditionDataset(Dataset):
    """All (A + B) % P pairs for given indices."""

    def __init__(self, indices):
        super().__init__()
        # indices into the flattened P*P grid
        self.pairs = [(i // P, i % P) for i in indices]

    def __getitem__(self, index):
        a, b = self.pairs[index]
        c = (a + b) % P
        x = np.array([a, b, EQ_TOKEN], dtype=np.float32)
        y = np.array([IGNORE, IGNORE, c], dtype=np.float32)
        return x, y

    def __len__(self):
        return len(self.pairs)
```

**Step 2: Add data splitting, model, and training loop**

```python
# ---------------------------------------------------------------------------
# Data split: 75% train, 25% test
# ---------------------------------------------------------------------------

np.random.seed(42)
all_indices = np.arange(P * P)              # 12,769 pairs
np.random.shuffle(all_indices)
split = int(0.75 * len(all_indices))         # 9,576 train
train_ds = ModularAdditionDataset(all_indices[:split])
test_ds  = ModularAdditionDataset(all_indices[split:])

BATCH_SIZE = 512
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_ds)} pairs | Test: {len(test_ds)} pairs")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = Transformer(
    vocab_size=VOCAB_SIZE,
    embed_dim=128,
    num_heads=4,
    ff_dim=512,
    num_layers=1,
    max_seq_len=SEQ_LEN,
    causal=True,
)
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = CrossEntropyLoss(ignore_index=IGNORE)

# ---------------------------------------------------------------------------
# Evaluation: accuracy on the answer position only
# ---------------------------------------------------------------------------

def evaluate(model, loader):
    """Return accuracy on the answer position (last token)."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        logits = model(x)                        # (B, 3, vocab)
        preds = logits.data[:, -1, :].argmax(axis=-1)  # last position
        targets = np.array(y)[:, -1].astype(int)
        correct += (preds == targets).sum()
        total += targets.shape[0]
    return correct / total

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

EPOCHS = 5000
print(f"\nTraining for up to {EPOCHS} epochs...")
print(f"{'Epoch':>6} | {'Loss':>10} | {'Train Acc':>10} | {'Test Acc':>10}")
print("-" * 50)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    num_batches = 0

    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        num_batches += 1

    # Log every 100 epochs (or first 10)
    if epoch <= 10 or epoch % 100 == 0:
        avg_loss = total_loss / num_batches
        train_acc = evaluate(model, train_loader)
        test_acc  = evaluate(model, test_loader)
        print(f"{epoch:>6} | {avg_loss:>10.4f} | {train_acc*100:>9.2f}% | {test_acc*100:>9.2f}%")

        # Early stop if both reach 100%
        if train_acc > 0.99 and test_acc > 0.99:
            print(f"\nGrokking achieved at epoch {epoch}!")
            break
```

**Step 3: Run it**

Run: `uv run python examples/modular_addition.py`
Expected: Starts printing epoch/loss/accuracy table. Train acc rises fast, test acc rises later.

**Step 4: Commit**

```bash
git add examples/modular_addition.py
git commit -m "Add modular addition grokking example"
```

### Task 2: Verify convergence and tune if needed

**Step 1: Run the full training**

Run: `uv run python examples/modular_addition.py`
Expected: Test accuracy eventually reaches ~100%.

If it doesn't converge:
- Try weight decay (add to Adam if not supported)
- Try lr=3e-4
- Try more epochs
- Try embed_dim=256

**Step 2: Adjust hyperparameters if needed and re-run**

**Step 3: Final commit with working hyperparameters**

```bash
git add examples/modular_addition.py
git commit -m "Tune modular addition hyperparameters for convergence"
```
