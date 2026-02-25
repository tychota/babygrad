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
