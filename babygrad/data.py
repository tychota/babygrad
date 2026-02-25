import gzip
import struct

import numpy as np


def parse_mnist(image_filename, label_filename):
    """Parse MNIST gzipped image and label files.

    Returns:
        Tuple (X, y):
            X (np.ndarray): Images as a (num_examples, 784) float32 array normalized to [0, 1].
            y (np.ndarray): Labels as a (num_examples,) uint8 array.
    """
    with gzip.open(image_filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows * cols)

    with gzip.open(label_filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images.astype(np.float32) / 255.0, labels


class Dataset:
    """Base class representing a dataset."""

    def __init__(self, transforms=None):
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def apply_transform(self, x):
        if not self.transforms:
            return x
        for t in self.transforms:
            x = t(x)
        return x


class MNISTDataset(Dataset):
    def __init__(self, image_filename, label_filename, transforms=None):
        super().__init__(transforms)
        self.images, self.labels = parse_mnist(image_filename, label_filename)

    def __getitem__(self, index):
        images = self.images[index]
        labels = self.labels[index]
        if isinstance(index, slice):
            images = images.reshape(-1, 28, 28, 1)
        else:
            images = images.reshape(28, 28, 1)
        images = self.apply_transform(images)
        return images, labels

    def __len__(self):
        return len(self.images)


class DataLoader:
    """Provides an iterator for easy batching, shuffling, and loading of data."""

    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        from .tensor import Tensor

        self._Tensor = Tensor
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.batch_idx = 0
        self.num_batches = len(self.dataset) // self.batch_size
        return self

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration
        start = self.batch_idx * self.batch_size
        batch_indices = self.indices[start : start + self.batch_size]
        samples = [self.dataset[i] for i in batch_indices]
        xs, ys = zip(*samples)
        self.batch_idx += 1
        return self._Tensor(np.stack(xs)), self._Tensor(np.stack(ys))
