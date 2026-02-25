import gzip
import struct

import numpy as np
import pytest

from babygrad.data import Dataset, DataLoader, MNISTDataset, parse_mnist
from babygrad.tensor import Tensor


def _write_mnist_images(path, images):
    """Write a small gzipped MNIST image file."""
    num, rows, cols = images.shape
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, num, rows, cols))
        f.write(images.astype(np.uint8).tobytes())


def _write_mnist_labels(path, labels):
    """Write a small gzipped MNIST label file."""
    with gzip.open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, len(labels)))
        f.write(np.array(labels, dtype=np.uint8).tobytes())


# ── Dataset base class ──────────────────────────────────────────────


class TestDatasetInit:
    def test_default_transforms_is_none(self):
        ds = Dataset()
        assert ds.transforms is None

    def test_accepts_transforms_list(self):
        transforms = [lambda x: x * 2]
        ds = Dataset(transforms=transforms)
        assert ds.transforms is transforms


class TestDatasetAbstractMethods:
    def test_getitem_raises_not_implemented(self):
        ds = Dataset()
        with pytest.raises(NotImplementedError):
            ds[0]

    def test_len_raises_not_implemented(self):
        ds = Dataset()
        with pytest.raises(NotImplementedError):
            len(ds)


class TestDatasetApplyTransform:
    def test_no_transforms_returns_input_unchanged(self):
        ds = Dataset()
        x = np.array([1.0, 2.0, 3.0])
        result = ds.apply_transform(x)
        np.testing.assert_array_equal(result, x)

    def test_single_transform_applied(self):
        ds = Dataset(transforms=[lambda x: x * 2])
        x = np.array([1.0, 2.0, 3.0])
        result = ds.apply_transform(x)
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_multiple_transforms_applied_in_order(self):
        ds = Dataset(transforms=[lambda x: x + 1, lambda x: x * 10])
        x = np.array([1.0, 2.0])
        result = ds.apply_transform(x)
        # (1+1)*10=20, (2+1)*10=30
        np.testing.assert_array_equal(result, [20.0, 30.0])


# ── parse_mnist ─────────────────────────────────────────────────────


class TestParseMnist:
    def test_returns_images_and_labels(self, tmp_path):
        images = np.zeros((3, 28, 28), dtype=np.uint8)
        labels = np.array([0, 1, 2], dtype=np.uint8)
        img_path = tmp_path / "images.gz"
        lbl_path = tmp_path / "labels.gz"
        _write_mnist_images(str(img_path), images)
        _write_mnist_labels(str(lbl_path), labels)

        X, y = parse_mnist(str(img_path), str(lbl_path))
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_images_shape_is_num_by_784(self, tmp_path):
        images = np.random.randint(0, 256, (5, 28, 28), dtype=np.uint8)
        labels = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        img_path = tmp_path / "images.gz"
        lbl_path = tmp_path / "labels.gz"
        _write_mnist_images(str(img_path), images)
        _write_mnist_labels(str(lbl_path), labels)

        X, y = parse_mnist(str(img_path), str(lbl_path))
        assert X.shape == (5, 784)

    def test_labels_shape(self, tmp_path):
        images = np.zeros((4, 28, 28), dtype=np.uint8)
        labels = np.array([7, 3, 1, 9], dtype=np.uint8)
        img_path = tmp_path / "images.gz"
        lbl_path = tmp_path / "labels.gz"
        _write_mnist_images(str(img_path), images)
        _write_mnist_labels(str(lbl_path), labels)

        X, y = parse_mnist(str(img_path), str(lbl_path))
        assert y.shape == (4,)
        np.testing.assert_array_equal(y, [7, 3, 1, 9])

    def test_images_normalized_to_float32(self, tmp_path):
        images = np.full((2, 28, 28), 255, dtype=np.uint8)
        labels = np.array([0, 1], dtype=np.uint8)
        img_path = tmp_path / "images.gz"
        lbl_path = tmp_path / "labels.gz"
        _write_mnist_images(str(img_path), images)
        _write_mnist_labels(str(lbl_path), labels)

        X, y = parse_mnist(str(img_path), str(lbl_path))
        assert X.dtype == np.float32
        np.testing.assert_allclose(X[0], np.ones(784, dtype=np.float32))

    def test_pixel_values_correctly_normalized(self, tmp_path):
        images = np.zeros((1, 28, 28), dtype=np.uint8)
        images[0, 0, 0] = 128
        labels = np.array([5], dtype=np.uint8)
        img_path = tmp_path / "images.gz"
        lbl_path = tmp_path / "labels.gz"
        _write_mnist_images(str(img_path), images)
        _write_mnist_labels(str(lbl_path), labels)

        X, y = parse_mnist(str(img_path), str(lbl_path))
        assert X[0, 0] == pytest.approx(128 / 255.0)


# ── MNISTDataset ────────────────────────────────────────────────────


@pytest.fixture
def mnist_files(tmp_path):
    """Create small MNIST fixture files (5 images, 28x28)."""
    np.random.seed(0)
    images = np.random.randint(0, 256, (5, 28, 28), dtype=np.uint8)
    labels = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    img_path = str(tmp_path / "images.gz")
    lbl_path = str(tmp_path / "labels.gz")
    _write_mnist_images(img_path, images)
    _write_mnist_labels(lbl_path, labels)
    return img_path, lbl_path


class TestMNISTDatasetLen:
    def test_len_returns_number_of_images(self, mnist_files):
        img, lbl = mnist_files
        ds = MNISTDataset(img, lbl)
        assert len(ds) == 5


class TestMNISTDatasetGetItemInt:
    def test_single_index_returns_tuple(self, mnist_files):
        img, lbl = mnist_files
        ds = MNISTDataset(img, lbl)
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_single_index_image_shape(self, mnist_files):
        img, lbl = mnist_files
        ds = MNISTDataset(img, lbl)
        image, label = ds[0]
        assert image.shape == (28, 28, 1)

    def test_single_index_label_value(self, mnist_files):
        img, lbl = mnist_files
        ds = MNISTDataset(img, lbl)
        _, label = ds[2]
        assert label == 2


class TestMNISTDatasetGetItemSlice:
    def test_slice_image_shape(self, mnist_files):
        img, lbl = mnist_files
        ds = MNISTDataset(img, lbl)
        images, labels = ds[1:4]
        assert images.shape == (3, 28, 28, 1)

    def test_slice_labels(self, mnist_files):
        img, lbl = mnist_files
        ds = MNISTDataset(img, lbl)
        _, labels = ds[0:3]
        np.testing.assert_array_equal(labels, [0, 1, 2])


class TestMNISTDatasetTransforms:
    def test_transform_applied_to_single_item(self, mnist_files):
        img, lbl = mnist_files
        ds = MNISTDataset(img, lbl, transforms=[lambda x: x * 0])
        image, _ = ds[0]
        np.testing.assert_array_equal(image, np.zeros((28, 28, 1)))


# ── DataLoader ──────────────────────────────────────────────────────


class NumberDataset(Dataset):
    """Simple dataset for testing DataLoader."""

    def __init__(self):
        super().__init__()
        self.x = np.arange(10, dtype=np.float32)
        self.y = np.arange(10, dtype=np.float32) * 2

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class TestDataLoaderIteration:
    def test_iterates_all_batches(self):
        ds = NumberDataset()  # 10 items
        loader = DataLoader(ds, batch_size=5, shuffle=False)
        batches = list(loader)
        assert len(batches) == 2

    def test_returns_tensor_tuples(self):
        ds = NumberDataset()
        loader = DataLoader(ds, batch_size=3, shuffle=False)
        x_batch, y_batch = next(iter(loader))
        assert isinstance(x_batch, Tensor)
        assert isinstance(y_batch, Tensor)

    def test_batch_shapes(self):
        ds = NumberDataset()
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        x_batch, y_batch = next(iter(loader))
        assert x_batch.shape == (4,)
        assert y_batch.shape == (4,)

    def test_batch_values_no_shuffle(self):
        ds = NumberDataset()
        loader = DataLoader(ds, batch_size=3, shuffle=False)
        x_batch, y_batch = next(iter(loader))
        np.testing.assert_array_equal(x_batch.data, [0, 1, 2])
        np.testing.assert_array_equal(y_batch.data, [0, 2, 4])

    def test_drops_incomplete_last_batch(self):
        ds = NumberDataset()  # 10 items, batch_size=3 -> 3 full batches (9 items)
        loader = DataLoader(ds, batch_size=3, shuffle=False)
        batches = list(loader)
        assert len(batches) == 3
        # Last batch has items at indices 9 only would be incomplete
        # 10 // 3 = 3 batches of 3 items = 9 items used

    def test_can_iterate_multiple_times(self):
        ds = NumberDataset()
        loader = DataLoader(ds, batch_size=5, shuffle=False)
        batches1 = list(loader)
        batches2 = list(loader)
        assert len(batches1) == len(batches2)
        np.testing.assert_array_equal(batches1[0][0].data, batches2[0][0].data)


class TestDataLoaderShuffle:
    def test_shuffle_changes_order(self):
        ds = NumberDataset()
        loader = DataLoader(ds, batch_size=10, shuffle=True)
        # Collect all items across iterations - at least one should differ
        np.random.seed(42)
        x1, _ = next(iter(loader))
        np.random.seed(99)
        x2, _ = next(iter(loader))
        # With different seeds, shuffled orders should (very likely) differ
        assert not np.array_equal(x1.data, x2.data)

    def test_no_shuffle_preserves_order(self):
        ds = NumberDataset()
        loader = DataLoader(ds, batch_size=5, shuffle=False)
        x_batch, _ = next(iter(loader))
        np.testing.assert_array_equal(x_batch.data, [0, 1, 2, 3, 4])
