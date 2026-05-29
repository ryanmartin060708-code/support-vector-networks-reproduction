"""Kernel functions used by the support-vector network implementation.

The 1995 paper uses the polynomial convolution

    K(u, v) = (u dot v + 1)^d

as its main experimental kernel. The module also includes linear and Gaussian
RBF kernels for completeness and for modern sanity checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable

import numpy as np


Array = np.ndarray
KernelCallable = Callable[[Array, Array], Array]


def _as_2d(x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 1D or 2D array, got shape {arr.shape}")
    return arr


def linear_kernel(x: Array, y: Array) -> Array:
    """Return the Gram matrix K(x, y) = x y^T."""

    x = _as_2d(x)
    y = _as_2d(y)
    return x @ y.T


def polynomial_kernel(
    x: Array,
    y: Array,
    degree: int = 2,
    coef0: float = 1.0,
    gamma: float = 1.0,
) -> Array:
    """Return K(x, y) = (gamma * x y^T + coef0)^degree.

    Cortes and Vapnik use gamma=1 and coef0=1. The explicit feature space
    contains monomials up to the requested degree when coef0 is non-zero.
    """

    if degree < 1:
        raise ValueError("polynomial degree must be >= 1")
    return (gamma * linear_kernel(x, y) + coef0) ** degree


def squared_euclidean_distances(x: Array, y: Array) -> Array:
    """Compute pairwise squared Euclidean distances with clipping."""

    x = _as_2d(x)
    y = _as_2d(y)
    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    distances = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(distances, 0.0)


def rbf_kernel(x: Array, y: Array, sigma: float = 1.0) -> Array:
    """Return K(x, y) = exp(-||x-y||^2 / (2 sigma^2))."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    distances = squared_euclidean_distances(x, y)
    return np.exp(-distances / (2.0 * sigma * sigma))


def polynomial_feature_dimension(
    n_features: int,
    degree: int,
    include_bias: bool = True,
) -> int:
    """Dimension of all monomials up to ``degree`` in ``n_features`` inputs.

    For K(x, y) = (x dot y + 1)^d, the canonical feature map contains all
    monomials of total degree <= d. The exact count is C(n+d, d), including a
    constant coordinate. The paper's Table 2 reports 256 rather than 257 for
    d=1; use include_bias=False to match that convention.
    """

    if n_features < 1:
        raise ValueError("n_features must be positive")
    if degree < 1:
        raise ValueError("degree must be >= 1")
    dim = comb(n_features + degree, degree)
    if not include_bias:
        dim -= 1
    return dim


@dataclass(frozen=True)
class KernelSpec:
    """Small serializable kernel descriptor."""

    name: str = "polynomial"
    degree: int = 2
    coef0: float = 1.0
    gamma: float = 1.0
    sigma: float = 1.0

    def make(self) -> KernelCallable:
        return make_kernel(
            self.name,
            degree=self.degree,
            coef0=self.coef0,
            gamma=self.gamma,
            sigma=self.sigma,
        )


def make_kernel(
    name: str,
    degree: int = 2,
    coef0: float = 1.0,
    gamma: float = 1.0,
    sigma: float = 1.0,
) -> KernelCallable:
    """Create a kernel callable from a short name."""

    normalized = name.lower().replace("-", "_")
    if normalized in {"linear", "dot", "dot_product"}:
        return linear_kernel
    if normalized in {"poly", "polynomial"}:
        return lambda x, y: polynomial_kernel(
            x, y, degree=degree, coef0=coef0, gamma=gamma
        )
    if normalized in {"rbf", "gaussian", "gaussian_rbf"}:
        return lambda x, y: rbf_kernel(x, y, sigma=sigma)
    raise ValueError(f"unknown kernel: {name}")
