"""Estimator classes for support-vector networks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .kernels import KernelCallable, KernelSpec, make_kernel
from .optimizer import (
    ChunkingOptimizer,
    DualQPSolver,
    SMOOptimizer,
    SVMOptimizationResult,
    validate_binary_labels,
)

Array = np.ndarray


@dataclass
class BinarySVM:
    """Binary support-vector machine trained in the dual.

    Parameters follow the paper where possible. Set ``C=None`` for hard margin
    training and a positive float for the L1 soft-margin box constraint.
    """

    C: float | None = 1.0
    kernel: str = "polynomial"
    degree: int = 2
    coef0: float = 1.0
    gamma: float = 1.0
    sigma: float = 1.0
    optimizer: str = "smo"
    tol: float = 1e-3
    eps: float = 1e-6
    max_iter: int = 10_000
    max_passes: int = 10
    random_state: int | None = 0
    store_kernel: bool = True

    classes_: Array | None = field(default=None, init=False)
    kernel_spec_: KernelSpec | None = field(default=None, init=False)
    kernel_fn_: KernelCallable | None = field(default=None, init=False)
    support_vectors_: Array | None = field(default=None, init=False)
    support_labels_: Array | None = field(default=None, init=False)
    support_alphas_: Array | None = field(default=None, init=False)
    alphas_: Array | None = field(default=None, init=False)
    bias_: float = field(default=0.0, init=False)
    result_: SVMOptimizationResult | None = field(default=None, init=False)
    x_train_: Array | None = field(default=None, init=False)
    y_train_: Array | None = field(default=None, init=False)
    w_: Array | None = field(default=None, init=False)

    def fit(self, x: Array, y: Array) -> "BinarySVM":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array")
        encoded_y = self._encode_labels(y)

        self.kernel_spec_ = KernelSpec(
            name=self.kernel,
            degree=self.degree,
            coef0=self.coef0,
            gamma=self.gamma,
            sigma=self.sigma,
        )
        self.kernel_fn_ = self.kernel_spec_.make()
        solver = self._make_solver()
        result = solver.solve(x, encoded_y, self.kernel_fn_, self.C)

        self.result_ = result
        self.alphas_ = result.alphas
        self.bias_ = result.bias
        support = result.support_indices
        self.support_vectors_ = x[support]
        self.support_labels_ = encoded_y[support]
        self.support_alphas_ = result.alphas[support]
        self.x_train_ = x
        self.y_train_ = encoded_y
        if self.kernel.lower() in {"linear", "dot", "dot_product"}:
            self.w_ = (result.alphas * encoded_y) @ x
        else:
            self.w_ = None
        return self

    def decision_function(self, x: Array) -> Array:
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        kernel_values = self.kernel_fn_(x, self.support_vectors_)
        return kernel_values @ (self.support_alphas_ * self.support_labels_) + self.bias_

    def predict(self, x: Array) -> Array:
        scores = self.decision_function(x)
        encoded = np.where(scores >= 0.0, 1.0, -1.0)
        if self.classes_ is None:
            return encoded
        return np.where(encoded > 0, self.classes_[1], self.classes_[0])

    def score(self, x: Array, y: Array) -> float:
        return float(np.mean(self.predict(x) == np.asarray(y)))

    def margin_width(self) -> float | None:
        """Return 2 / ||w|| for linear kernels, otherwise None."""

        if self.w_ is None:
            return None
        norm = float(np.linalg.norm(self.w_))
        if norm == 0.0:
            return None
        return 2.0 / norm

    def support_count(self) -> int:
        self._check_fitted()
        return int(self.support_alphas_.size)

    def get_diagnostics(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "support_vectors": self.support_count(),
            "bias": self.bias_,
            "converged": self.result_.converged,
            "iterations": self.result_.n_iter,
            "kkt": self.result_.kkt_violations,
            "optimizer": self.result_.diagnostics.get("optimizer", self.optimizer),
            "dual_objective": self.result_.diagnostics.get("objective"),
        }

    def _encode_labels(self, y: Array) -> Array:
        arr = np.asarray(y).ravel()
        unique = np.unique(arr)
        if unique.size != 2:
            raise ValueError(f"BinarySVM requires exactly two classes, got {unique}")
        try:
            numeric_values = set(unique.astype(float).tolist())
        except (TypeError, ValueError):
            numeric_values = set()
        if numeric_values == {-1.0, 1.0}:
            self.classes_ = np.array([-1.0, 1.0])
            return validate_binary_labels(arr.astype(float))
        self.classes_ = unique
        return np.where(arr == unique[1], 1.0, -1.0)

    def _make_solver(self) -> SMOOptimizer | DualQPSolver | ChunkingOptimizer:
        name = self.optimizer.lower()
        if name == "auto":
            name = "qp" if self.C is None else "smo"
        if name == "smo":
            return SMOOptimizer(
                max_iter=self.max_iter,
                max_passes=self.max_passes,
                tol=self.tol,
                eps=self.eps,
                random_state=self.random_state,
                store_kernel=self.store_kernel,
            )
        if name == "qp":
            return DualQPSolver(max_iter=self.max_iter, tol=self.tol)
        if name == "chunking":
            return ChunkingOptimizer(
                base_optimizer=SMOOptimizer(
                    max_iter=self.max_iter,
                    max_passes=self.max_passes,
                    tol=self.tol,
                    eps=self.eps,
                    random_state=self.random_state,
                    store_kernel=self.store_kernel,
                ),
                tol=self.tol,
            )
        raise ValueError(f"unknown optimizer: {self.optimizer}")

    def _check_fitted(self) -> None:
        if self.support_vectors_ is None or self.kernel_fn_ is None:
            raise RuntimeError("model is not fitted")


class HardMarginSVM(BinarySVM):
    """Hard-margin SVM with alpha_i >= 0 and no upper box constraint."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("C", None)
        kwargs.setdefault("optimizer", "auto")
        super().__init__(**kwargs)


class SoftMarginSVM(BinarySVM):
    """Soft-margin SVM with L1 slack penalty C sum_i xi_i."""

    def __init__(self, C: float = 1.0, **kwargs: Any) -> None:
        if C <= 0:
            raise ValueError("C must be positive")
        super().__init__(C=C, **kwargs)


@dataclass
class OneVsRestSVM:
    """One-vs-rest multiclass support-vector network.

    This matches the digit experiments in the paper: ten binary classifiers are
    trained and prediction uses the class with maximum raw output.
    """

    base_estimator: BinarySVM = field(default_factory=SoftMarginSVM)
    classes_: Array | None = field(default=None, init=False)
    estimators_: list[BinarySVM] = field(default_factory=list, init=False)

    def fit(self, x: Array, y: Array) -> "OneVsRestSVM":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for cls in self.classes_:
            binary_y = np.where(y == cls, 1.0, -1.0)
            estimator = deepcopy(self.base_estimator)
            estimator.fit(x, binary_y)
            self.estimators_.append(estimator)
        return self

    def decision_function(self, x: Array) -> Array:
        if not self.estimators_:
            raise RuntimeError("model is not fitted")
        return np.column_stack([est.decision_function(x) for est in self.estimators_])

    def predict(self, x: Array) -> Array:
        if self.classes_ is None:
            raise RuntimeError("model is not fitted")
        scores = self.decision_function(x)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, x: Array, y: Array) -> float:
        return float(np.mean(self.predict(x) == np.asarray(y).ravel()))

    def support_counts(self) -> Array:
        if not self.estimators_:
            raise RuntimeError("model is not fitted")
        return np.array([est.support_count() for est in self.estimators_], dtype=int)

    def diagnostics(self) -> list[dict[str, Any]]:
        return [est.get_diagnostics() for est in self.estimators_]
