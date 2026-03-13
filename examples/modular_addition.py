"""
Modular Addition with a Transformer (Grokking)
================================================
Train a single-layer Transformer to learn (A + B) % 113.

This demonstrates the "grokking" phenomenon: the model memorizes the
training set quickly (train accuracy -> 100%) but takes much longer to
generalize (test accuracy -> 100%).

Vocabulary: 0-112 = numbers, 113 = "=" token
Input:  [A, B, =]   (3 tokens)
Target: [-1, -1, C]  (loss only on answer position, C = (A+B) % 113)
"""

import numpy as np
from babygrad import (
    Tensor, Transformer, CrossEntropyLoss, Adam, Dataset, DataLoader, Trainer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

P = 113            # prime modulus
EQ_TOKEN = P       # "=" token has id 113
VOCAB_SIZE = P + 1 # 0..112 + "="
SEQ_LEN = 3       # [A, B, =]
IGNORE = -1        # ignore index for loss

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ModularAdditionDataset(Dataset):
    """All (A + B) % P pairs for given indices into the P*P grid."""

    def __init__(self, indices):
        super().__init__()
        self.pairs = [(i // P, i % P) for i in indices]

    def __getitem__(self, index):
        a, b = self.pairs[index]
        c = (a + b) % P
        x = np.array([a, b, EQ_TOKEN], dtype=np.float32)
        y = np.array([IGNORE, IGNORE, c], dtype=np.float32)
        return x, y

    def __len__(self):
        return len(self.pairs)

# ---------------------------------------------------------------------------
# Data split: 75% train, 25% test
# ---------------------------------------------------------------------------

np.random.seed(42)
all_indices = np.arange(P * P)                # 12,769 pairs
np.random.shuffle(all_indices)
split = int(0.75 * len(all_indices))           # 9,576 train

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
# Custom evaluation: accuracy on the answer position only
# ---------------------------------------------------------------------------

def answer_accuracy(model, loader):
    """Accuracy on the last position (the answer token)."""
    correct = 0
    total = 0
    for x, y in loader:
        logits = model(x)                             # (B, 3, vocab)
        preds = logits.data[:, -1, :].argmax(axis=-1) # last position
        targets = np.array(y)[:, -1].astype(int)
        correct += (preds == targets).sum()
        total += targets.shape[0]
    return correct / total

# ---------------------------------------------------------------------------
# Training with Trainer (no val_loader to avoid auto-eval every epoch)
# ---------------------------------------------------------------------------

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    compute_metrics=answer_accuracy,
)

EPOCHS = 5000
print(f"\nTraining for up to {EPOCHS} epochs...\n")

for epoch in range(1, EPOCHS + 1):
    trainer.fit(1)

    # Detailed logging every 100 epochs (or first 10)
    if epoch <= 10 or epoch % 100 == 0:
        train_acc = trainer.evaluate(loader=train_loader)
        test_acc  = trainer.evaluate(loader=test_loader)
        print(f"  -> Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

        if train_acc > 0.99 and test_acc > 0.99:
            print(f"\nGrokking achieved at epoch {epoch}!")
            break
