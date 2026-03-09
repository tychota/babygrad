from babygrad.tensor import Tensor


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

    def fit(self, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.data
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f}", end="")

            if self.val_loader is not None:
                val_acc = self.evaluate()
                print(f" | Val Acc: {val_acc * 100:.2f}%")
            else:
                print()
