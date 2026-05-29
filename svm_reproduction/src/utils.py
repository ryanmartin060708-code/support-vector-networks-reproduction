"""Utilities for reproducible experiments and digit preprocessing."""

from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import ssl
import struct
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

Array = np.ndarray


USPS_URLS = {
    "train": [
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps",
    ],
    "test": [
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t",
    ],
}

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def set_random_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(
    urls: str | Iterable[str],
    destination: str | Path,
    sha256: str | None = None,
    timeout: int = 60,
) -> Path:
    """Download a file, trying mirrors in order."""

    destination = Path(destination)
    ensure_dir(destination.parent)
    if isinstance(urls, str):
        urls = [urls]
    last_error: Exception | None = None
    for url in urls:
        try:
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    data = response.read()
            except Exception as tls_exc:
                if "CERTIFICATE_VERIFY_FAILED" not in str(tls_exc):
                    raise
                context = ssl._create_unverified_context()
                with urllib.request.urlopen(
                    url, timeout=timeout, context=context
                ) as response:
                    data = response.read()
            if sha256 is not None:
                digest = hashlib.sha256(data).hexdigest()
                if digest != sha256:
                    raise ValueError(
                        f"sha256 mismatch for {url}: expected {sha256}, got {digest}"
                    )
            destination.write_bytes(data)
            return destination
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc
    raise RuntimeError(f"failed to download {destination.name}: {last_error}")


def load_libsvm(path: str | Path, n_features: int | None = None) -> tuple[Array, Array]:
    """Load a dense array from a libsvm-format file, including .bz2/.gz."""

    path = Path(path)
    opener = open
    if path.suffix == ".bz2":
        opener = bz2.open
    elif path.suffix == ".gz":
        opener = gzip.open

    labels: list[float] = []
    rows: list[dict[int, float]] = []
    inferred_features = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            labels.append(float(parts[0]))
            row: dict[int, float] = {}
            for item in parts[1:]:
                index, value = item.split(":", 1)
                zero_index = int(index) - 1
                row[zero_index] = float(value)
                inferred_features = max(inferred_features, zero_index + 1)
            rows.append(row)

    if n_features is None:
        n_features = inferred_features
    x = np.zeros((len(rows), n_features), dtype=float)
    for row_idx, row in enumerate(rows):
        for feature_idx, value in row.items():
            if feature_idx < n_features:
                x[row_idx, feature_idx] = value
    y = normalize_digit_labels(np.asarray(labels))
    return x, y


def normalize_digit_labels(labels: Array) -> Array:
    """Map common digit label conventions to 0..9."""

    labels = np.asarray(labels)
    unique = set(labels.astype(int).tolist())
    if unique == set(range(1, 11)):
        return (labels.astype(int) % 10).astype(int)
    return labels.astype(int)


def fetch_usps(data_dir: str | Path) -> tuple[Path, Path]:
    data_dir = ensure_dir(data_dir)
    train_path = data_dir / "usps.bz2"
    test_path = data_dir / "usps.t.bz2"
    if not train_path.exists():
        download_file(USPS_URLS["train"], train_path)
    if not test_path.exists():
        download_file(USPS_URLS["test"], test_path)
    return train_path, test_path


def load_usps(data_dir: str | Path, download: bool = True) -> tuple[Array, Array, Array, Array]:
    data_dir = ensure_dir(data_dir)
    train_path = data_dir / "usps.bz2"
    test_path = data_dir / "usps.t.bz2"
    if download and (not train_path.exists() or not test_path.exists()):
        fetch_usps(data_dir)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "USPS files not found. Run experiments/usps.py with --download or "
            "place usps.bz2 and usps.t.bz2 in the data directory."
        )
    x_train, y_train = load_libsvm(train_path, n_features=256)
    x_test, y_test = load_libsvm(test_path, n_features=256)
    return x_train, y_train, x_test, y_test


def fetch_mnist(data_dir: str | Path) -> dict[str, Path]:
    data_dir = ensure_dir(data_dir)
    paths = {}
    for name, url in MNIST_URLS.items():
        path = data_dir / f"{name}.gz"
        if not path.exists():
            download_file(url, path)
        paths[name] = path
    return paths


def read_idx_images(path: str | Path) -> Array:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"invalid IDX image magic {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols).astype(float) / 255.0


def read_idx_labels(path: str | Path) -> Array:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"invalid IDX label magic {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if data.size != count:
        raise ValueError("IDX label count mismatch")
    return data.astype(int)


def load_mnist(data_dir: str | Path, download: bool = True) -> tuple[Array, Array, Array, Array]:
    data_dir = ensure_dir(data_dir)
    paths = {name: data_dir / f"{name}.gz" for name in MNIST_URLS}
    if download and not all(path.exists() for path in paths.values()):
        paths = fetch_mnist(data_dir)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "MNIST files not found. Run experiments/mnist.py with --download or "
            f"place IDX gzip files in {data_dir}. Missing: {missing}"
        )
    x_train = read_idx_images(paths["train_images"]).reshape(-1, 28 * 28)
    y_train = read_idx_labels(paths["train_labels"])
    x_test = read_idx_images(paths["test_images"]).reshape(-1, 28 * 28)
    y_test = read_idx_labels(paths["test_labels"])
    return x_train, y_train, x_test, y_test


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> Array:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)


def gaussian_smooth_images(images: Array, sigma: float = 0.75) -> Array:
    """Apply separable Gaussian smoothing without SciPy."""

    if sigma <= 0:
        return np.asarray(images, dtype=float)
    arr = np.asarray(images, dtype=float)
    if arr.ndim == 2:
        side = int(np.sqrt(arr.shape[1]))
        arr = arr.reshape(-1, side, side)
    kernel = gaussian_kernel_1d(sigma)
    radius = kernel.size // 2
    padded = np.pad(arr, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    temp = np.zeros_like(arr)
    for offset, weight in enumerate(kernel):
        temp += weight * padded[:, :, offset : offset + arr.shape[2]]
    padded = np.pad(temp, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for offset, weight in enumerate(kernel):
        out += weight * padded[:, offset : offset + arr.shape[1], :]
    return out


def _shift_image_integer(image: Array, shift_y: int, shift_x: int) -> Array:
    out = np.zeros_like(image)
    h, w = image.shape
    src_y0 = max(0, -shift_y)
    src_y1 = min(h, h - shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(w, w - shift_x)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(h, h + shift_y)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(w, w + shift_x)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def center_images(images: Array) -> Array:
    """Center each image by its intensity center of mass."""

    arr = np.asarray(images, dtype=float)
    flat_input = arr.ndim == 2
    if flat_input:
        side = int(np.sqrt(arr.shape[1]))
        arr = arr.reshape(-1, side, side)
    n, h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    centered = np.zeros_like(arr)
    target_y = (h - 1) / 2.0
    target_x = (w - 1) / 2.0
    for idx in range(n):
        image = arr[idx]
        mass = float(np.sum(image))
        if mass <= 1e-12:
            centered[idx] = image
            continue
        cy = float(np.sum(yy * image) / mass)
        cx = float(np.sum(xx * image) / mass)
        centered[idx] = _shift_image_integer(
            image, int(round(target_y - cy)), int(round(target_x - cx))
        )
    return centered.reshape(arr.shape[0], -1) if flat_input else centered


def deslant_images(images: Array) -> Array:
    """Deskew digits using an intensity-moment horizontal shear."""

    arr = np.asarray(images, dtype=float)
    flat_input = arr.ndim == 2
    if flat_input:
        side = int(np.sqrt(arr.shape[1]))
        arr = arr.reshape(-1, side, side)
    n, h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros_like(arr)
    for idx in range(n):
        image = arr[idx]
        mass = float(np.sum(image))
        if mass <= 1e-12:
            out[idx] = image
            continue
        cy = float(np.sum(yy * image) / mass)
        cx = float(np.sum(xx * image) / mass)
        y_centered = yy - cy
        x_centered = xx - cx
        var_y = float(np.sum((y_centered**2) * image) / mass)
        cov_xy = float(np.sum((x_centered * y_centered) * image) / mass)
        skew = cov_xy / (var_y + 1e-12)
        source_x = xx - skew * y_centered
        x0 = np.floor(source_x).astype(int)
        x1 = x0 + 1
        weight = source_x - x0
        valid0 = (x0 >= 0) & (x0 < w)
        valid1 = (x1 >= 0) & (x1 < w)
        sampled = np.zeros_like(image)
        sampled[valid0] += (1.0 - weight[valid0]) * image[yy[valid0], x0[valid0]]
        sampled[valid1] += weight[valid1] * image[yy[valid1], x1[valid1]]
        out[idx] = sampled
    return out.reshape(arr.shape[0], -1) if flat_input else out


def preprocess_usps(x: Array, smooth_sigma: float = 0.75) -> Array:
    """Center, deslant, and smooth USPS images as described in the paper."""

    images = np.asarray(x, dtype=float).reshape(-1, 16, 16)
    # LibSVM USPS values are commonly in [-1, 1]. Use nonnegative intensities
    # for moment computations, then keep normalized output in [0, 1].
    images = (images - images.min(axis=(1, 2), keepdims=True)) / (
        np.ptp(images, axis=(1, 2), keepdims=True) + 1e-12
    )
    images = center_images(images)
    images = deslant_images(images)
    images = gaussian_smooth_images(images, sigma=smooth_sigma)
    return images.reshape(images.shape[0], -1)


def subsample_per_class(
    x: Array,
    y: Array,
    max_per_class: int | None,
    seed: int | None = 0,
) -> tuple[Array, Array]:
    if max_per_class is None:
        return np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    indices = []
    y = np.asarray(y)
    for cls in np.unique(y):
        cls_indices = np.flatnonzero(y == cls)
        rng.shuffle(cls_indices)
        indices.extend(cls_indices[:max_per_class].tolist())
    indices = np.array(sorted(indices))
    return np.asarray(x)[indices], y[indices]


def standardize(
    x_train: Array,
    x_test: Array | None = None,
    eps: float = 1e-12,
) -> tuple[Array, Array | None]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_std = (x_train - mean) / (std + eps)
    if x_test is None:
        return x_train_std, None
    return x_train_std, (x_test - mean) / (std + eps)


def write_json(path: str | Path, payload: object) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
