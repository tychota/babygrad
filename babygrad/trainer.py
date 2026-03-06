from babygrad.tensor import Tensor


class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
