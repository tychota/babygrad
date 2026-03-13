# Modular Addition Example Design

## Task
Learn `(A + B) % 113` where A,B in [0,112].
Input sequence: `[A, B, =]` (3 tokens) → predict answer `C` at last position.

## Vocabulary
114 tokens: 0-112 = numbers, 113 = `=` sign.

## Data
- Enumerate all 113 x 113 = 12,769 pairs
- Random 75/25 split → ~9,577 train / ~3,192 test
- Input: `[A, B, 113]`, target: `[-1, -1, C]`
- `CrossEntropyLoss(ignore_index=-1)` — loss only on answer position

## Model
`Transformer(vocab_size=114, embed_dim=128, num_heads=4, ff_dim=512, num_layers=1, max_seq_len=4, causal=True)`

## Training
- Optimizer: Adam, lr=1e-3
- Batch size: 512
- Train until test accuracy reaches ~100% (grokking)

## Evaluation
Accuracy only on the answer position (argmax of last token logits vs true C).

## File
`examples/modular_addition.py` — readable, commented, prints train loss + test accuracy per epoch.

## Success Criteria
- Train accuracy reaches ~100% quickly
- Test accuracy eventually reaches ~100% (grokking phenomenon)
