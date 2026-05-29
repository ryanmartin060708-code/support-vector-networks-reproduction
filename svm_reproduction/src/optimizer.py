"""Dual optimizers for support-vector machines.

The implementation follows the dual problems in Cortes and Vapnik:

Hard margin:
    maximize 1^T alpha - 1/2 alpha^T Q alpha
    subject to alpha >= 0 and y^T alpha = 0.

L1 soft margin:
    same objective, with 0 <= alpha <= C and y^T alpha = 0.

The paper's appendix also derives variants for squared slack penalties. The
production estimator in ``svm.py`` uses the L1 box-constrained formulation
because it is the objective requested in the reproduction task and is now the
standard soft-margin SVM form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Array = np.ndarray
KernelCallable = Callable[[Array, Array], Array]


@dataclass
class SVMOptimizationResult:
    """Container returned by dual optimizers."""

    alphas: Array
    bias: float
    objective_history: list[float]
    n_iter: int
    converged: bool
    support_indices: Array
    kkt_violations: dict[str, float]
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)


def validate_binary_labels(y: Array) -> Array:
    y = np.asarray(y, dtype=float).ravel()
    values = set(np.unique(y).tolist())
    if not values.issubset({-1.0, 1.0}):
        raise ValueError(f"labels must be encoded as -1/+1, got {sorted(values)}")
    return y


def dual_objective(alphas: Array, y: Array, gram: Array) -> float:
    """Compute W(alpha) = sum alpha_i - 1/2 alpha^T Q alpha."""

    alphas = np.asarray(alphas, dtype=float).ravel()
    y = validate_binary_labels(y)
    q = (y[:, None] * y[None, :]) * gram
    return float(np.sum(alphas) - 0.5 * alphas @ q @ alphas)


def decision_from_gram(alphas: Array, y: Array, gram: Array, bias: float) -> Array:
    """Evaluate f_i = sum_j alpha_j y_j K(x_j, x_i) + b on training data."""

    return gram @ (alphas * y) + bias


def compute_bias(
    alphas: Array,
    y: Array,
    gram: Array,
    C: float | None,
    eps: float = 1e-8,
) -> float:
    """Recover the intercept from KKT conditions.

    For free support vectors, y_i f(x_i) = 1, so
    b = y_i - sum_j alpha_j y_j K(x_j, x_i).
    """

    alphas = np.asarray(alphas, dtype=float).ravel()
    y = validate_binary_labels(y)
    if C is None or np.isinf(C):
        free = alphas > eps
    else:
        free = (alphas > eps) & (alphas < C - eps)
    support = alphas > eps
    candidates = np.where(free)[0]
    if candidates.size == 0:
        candidates = np.where(support)[0]
    if candidates.size == 0:
        return 0.0
    weighted = gram[:, candidates].T @ (alphas * y)
    return float(np.mean(y[candidates] - weighted))


def compute_bias_from_kernel(
    x: Array,
    y: Array,
    alphas: Array,
    kernel: KernelCallable,
    C: float | None,
    eps: float = 1e-8,
    batch_size: int = 512,
) -> float:
    """Recover bias without building the full Gram matrix."""

    if C is None or np.isinf(C):
        candidates = np.flatnonzero(alphas > eps)
    else:
        candidates = np.flatnonzero((alphas > eps) & (alphas < C - eps))
    if candidates.size == 0:
        candidates = np.flatnonzero(alphas > eps)
    if candidates.size == 0:
        return 0.0

    weighted = alphas * y
    values = []
    for start in range(0, candidates.size, batch_size):
        idx = candidates[start : start + batch_size]
        scores = kernel(x[idx], x) @ weighted
        values.extend((y[idx] - scores).tolist())
    return float(np.mean(values))


def dual_objective_from_support(
    x: Array,
    y: Array,
    alphas: Array,
    kernel: KernelCallable,
) -> float:
    """Compute the dual objective using only nonzero-alpha rows."""

    support = np.flatnonzero(alphas > 0.0)
    if support.size == 0:
        return 0.0
    a = alphas[support]
    yy = y[support]
    gram = kernel(x[support], x[support])
    q = (yy[:, None] * yy[None, :]) * gram
    return float(np.sum(a) - 0.5 * a @ q @ a)


def kkt_violation_summary_from_kernel(
    x: Array,
    y: Array,
    alphas: Array,
    kernel: KernelCallable,
    bias: float,
    C: float | None,
    eps: float = 1e-6,
    batch_size: int = 512,
) -> dict[str, float]:
    """Summarize KKT violations in batches without a full Gram matrix."""

    weighted = alphas * y
    violations = []
    finite_c = C is not None and not np.isinf(C)
    for start in range(0, x.shape[0], batch_size):
        stop = min(x.shape[0], start + batch_size)
        scores = kernel(x[start:stop], x) @ weighted + bias
        margins = y[start:stop] * scores
        alpha_batch = alphas[start:stop]
        batch = np.zeros_like(margins)
        if finite_c:
            at_lower = alpha_batch <= eps
            at_upper = alpha_batch >= float(C) - eps
            free = ~(at_lower | at_upper)
            batch[at_lower] = np.maximum(0.0, 1.0 - margins[at_lower])
            batch[free] = np.abs(margins[free] - 1.0)
            batch[at_upper] = np.maximum(0.0, margins[at_upper] - 1.0)
        else:
            at_lower = alpha_batch <= eps
            free = ~at_lower
            batch[at_lower] = np.maximum(0.0, 1.0 - margins[at_lower])
            batch[free] = np.abs(margins[free] - 1.0)
        violations.append(batch)
    if not violations:
        return {"max": 0.0, "mean": 0.0, "rms": 0.0}
    all_violations = np.concatenate(violations)
    return {
        "max": float(np.max(all_violations)),
        "mean": float(np.mean(all_violations)),
        "rms": float(np.sqrt(np.mean(all_violations * all_violations))),
    }


def kkt_violation_summary(
    alphas: Array,
    y: Array,
    gram: Array,
    bias: float,
    C: float | None,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Summarize absolute KKT residuals for margin conditions."""

    alphas = np.asarray(alphas, dtype=float).ravel()
    y = validate_binary_labels(y)
    margins = y * decision_from_gram(alphas, y, gram, bias)
    violations = np.zeros_like(margins)

    if C is None or np.isinf(C):
        at_lower = alphas <= eps
        free = ~at_lower
        violations[at_lower] = np.maximum(0.0, 1.0 - margins[at_lower])
        violations[free] = np.abs(margins[free] - 1.0)
    else:
        at_lower = alphas <= eps
        at_upper = alphas >= C - eps
        free = ~(at_lower | at_upper)
        violations[at_lower] = np.maximum(0.0, 1.0 - margins[at_lower])
        violations[free] = np.abs(margins[free] - 1.0)
        violations[at_upper] = np.maximum(0.0, margins[at_upper] - 1.0)

    return {
        "max": float(np.max(violations)) if violations.size else 0.0,
        "mean": float(np.mean(violations)) if violations.size else 0.0,
        "rms": float(np.sqrt(np.mean(violations * violations))) if violations.size else 0.0,
    }


class DualQPSolver:
    """Solve the exact dual QP with SciPy when available.

    This is useful for small problems and for hard-margin training. It is not
    intended for the full USPS/MNIST reproductions, where the from-scratch SMO
    optimizer is more memory- and time-conscious.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-8,
        support_tol: float = 1e-7,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.support_tol = support_tol

    def solve(
        self,
        x: Array,
        y: Array,
        kernel: KernelCallable,
        C: float | None = None,
    ) -> SVMOptimizationResult:
        try:
            from scipy.optimize import minimize
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "DualQPSolver requires scipy. Install project requirements or "
                "use optimizer='smo'."
            ) from exc

        x = np.asarray(x, dtype=float)
        y = validate_binary_labels(y)
        n_samples = x.shape[0]
        gram = kernel(x, x)
        q = (y[:, None] * y[None, :]) * gram

        def primalized_objective(alpha: Array) -> float:
            return float(0.5 * alpha @ q @ alpha - np.sum(alpha))

        def jac(alpha: Array) -> Array:
            return q @ alpha - np.ones_like(alpha)

        constraints = [{"type": "eq", "fun": lambda a: float(y @ a), "jac": lambda a: y}]
        upper = None if C is None or np.isinf(C) else float(C)
        bounds = [(0.0, upper) for _ in range(n_samples)]

        # A feasible zero vector is a reliable starting point.
        result = minimize(
            primalized_objective,
            np.zeros(n_samples, dtype=float),
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": self.tol, "maxiter": self.max_iter, "disp": False},
        )
        alphas = np.maximum(result.x, 0.0)
        if upper is not None:
            alphas = np.minimum(alphas, upper)
        bias = compute_bias(alphas, y, gram, C, eps=self.support_tol)
        support = np.flatnonzero(alphas > self.support_tol)
        objective = dual_objective(alphas, y, gram)
        return SVMOptimizationResult(
            alphas=alphas,
            bias=bias,
            objective_history=[objective],
            n_iter=int(result.nit),
            converged=bool(result.success),
            support_indices=support,
            kkt_violations=kkt_violation_summary(
                alphas, y, gram, bias, C, eps=self.support_tol
            ),
            diagnostics={
                "message": str(result.message),
                "optimizer": "scipy-slsqp",
                "objective": objective,
            },
        )


class SMOOptimizer:
    """Sequential minimal optimization implemented from scratch."""

    def __init__(
        self,
        max_iter: int = 10_000,
        max_passes: int = 10,
        tol: float = 1e-3,
        eps: float = 1e-6,
        support_tol: float = 1e-6,
        random_state: int | None = 0,
        store_kernel: bool = True,
    ) -> None:
        self.max_iter = max_iter
        self.max_passes = max_passes
        self.tol = tol
        self.eps = eps
        self.support_tol = support_tol
        self.random_state = random_state
        self.store_kernel = store_kernel

    def solve(
        self,
        x: Array,
        y: Array,
        kernel: KernelCallable,
        C: float | None = 1.0,
    ) -> SVMOptimizationResult:
        x = np.asarray(x, dtype=float)
        y = validate_binary_labels(y)
        n_samples = x.shape[0]
        if n_samples < 2:
            raise ValueError("SMO requires at least two training samples")

        finite_c = C is not None and not np.isinf(C)
        c_value = float(C) if finite_c else np.inf
        rng = np.random.default_rng(self.random_state)

        if self.store_kernel:
            gram = kernel(x, x)

            def k_col(i: int) -> Array:
                return gram[:, i]

            def k_val(i: int, j: int) -> float:
                return float(gram[i, j])

        else:
            # Memory-saving path. The final diagnostics still require a Gram
            # matrix and therefore are only suitable for moderate data sizes.
            gram_cache: dict[int, Array] = {}

            def k_col(i: int) -> Array:
                if i not in gram_cache:
                    gram_cache[i] = kernel(x, x[i : i + 1]).ravel()
                return gram_cache[i]

            def k_val(i: int, j: int) -> float:
                return float(k_col(j)[i])

            gram = None

        alphas = np.zeros(n_samples, dtype=float)
        bias = 0.0
        decision = np.zeros(n_samples, dtype=float)
        errors = decision - y
        objective_history: list[float] = []
        passes_without_change = 0
        n_iter = 0

        while passes_without_change < self.max_passes and n_iter < self.max_iter:
            num_changed = 0
            for i in rng.permutation(n_samples):
                e_i = errors[i]
                y_e_i = y[i] * e_i
                can_increase = (not finite_c) or (alphas[i] < c_value - self.eps)
                can_decrease = alphas[i] > self.eps
                violates = (y_e_i < -self.tol and can_increase) or (
                    y_e_i > self.tol and can_decrease
                )
                if not violates:
                    continue

                j = self._choose_second_index(i, errors, alphas, c_value, rng, finite_c)
                if j == i:
                    continue

                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]
                e_j = errors[j]

                if y[i] != y[j]:
                    low = max(0.0, alpha_j_old - alpha_i_old)
                    high = c_value + alpha_j_old - alpha_i_old if finite_c else np.inf
                else:
                    low = (
                        max(0.0, alpha_i_old + alpha_j_old - c_value)
                        if finite_c
                        else 0.0
                    )
                    high = alpha_i_old + alpha_j_old
                if high <= low + self.eps:
                    continue

                k_ii = k_val(i, i)
                k_jj = k_val(j, j)
                k_ij = k_val(i, j)
                eta = 2.0 * k_ij - k_ii - k_jj
                if eta >= -1e-14:
                    continue

                alpha_j_new = alpha_j_old - y[j] * (e_i - e_j) / eta
                alpha_j_new = max(low, alpha_j_new)
                if np.isfinite(high):
                    alpha_j_new = min(high, alpha_j_new)
                if abs(alpha_j_new - alpha_j_old) < self.eps * (
                    alpha_j_new + alpha_j_old + self.eps
                ):
                    continue

                alpha_i_new = alpha_i_old + y[i] * y[j] * (
                    alpha_j_old - alpha_j_new
                )
                if alpha_i_new < 0.0 and alpha_i_new > -1e-8:
                    alpha_i_new = 0.0
                if finite_c and alpha_i_new > c_value and alpha_i_new < c_value + 1e-8:
                    alpha_i_new = c_value
                if alpha_i_new < -1e-8:
                    continue
                if finite_c and alpha_i_new > c_value + 1e-8:
                    continue

                b1 = (
                    bias
                    - e_i
                    - y[i] * (alpha_i_new - alpha_i_old) * k_ii
                    - y[j] * (alpha_j_new - alpha_j_old) * k_ij
                )
                b2 = (
                    bias
                    - e_j
                    - y[i] * (alpha_i_new - alpha_i_old) * k_ij
                    - y[j] * (alpha_j_new - alpha_j_old) * k_jj
                )

                if self._is_free(alpha_i_new, c_value, finite_c):
                    new_bias = b1
                elif self._is_free(alpha_j_new, c_value, finite_c):
                    new_bias = b2
                else:
                    new_bias = 0.5 * (b1 + b2)

                alphas[i] = alpha_i_new
                alphas[j] = alpha_j_new
                delta_i = y[i] * (alpha_i_new - alpha_i_old)
                delta_j = y[j] * (alpha_j_new - alpha_j_old)
                decision += delta_i * k_col(i) + delta_j * k_col(j) + (
                    new_bias - bias
                )
                bias = float(new_bias)
                errors = decision - y
                num_changed += 1

            n_iter += 1
            if self.store_kernel:
                objective_history.append(dual_objective(alphas, y, gram))
            if num_changed == 0:
                passes_without_change += 1
            else:
                passes_without_change = 0

        support = np.flatnonzero(alphas > self.support_tol)
        if gram is None:
            bias = compute_bias_from_kernel(
                x, y, alphas, kernel, C, eps=self.support_tol
            )
            objective = dual_objective_from_support(x, y, alphas, kernel)
            violations = kkt_violation_summary_from_kernel(
                x, y, alphas, kernel, bias, C, eps=self.support_tol
            )
        else:
            bias = compute_bias(alphas, y, gram, C, eps=self.support_tol)
            objective = dual_objective(alphas, y, gram)
            violations = kkt_violation_summary(
                alphas, y, gram, bias, C, eps=self.support_tol
            )
        if not objective_history or objective_history[-1] != objective:
            objective_history.append(objective)
        converged = bool(
            passes_without_change >= self.max_passes
            or violations["max"] <= max(10.0 * self.tol, 1e-5)
        )
        return SVMOptimizationResult(
            alphas=alphas,
            bias=bias,
            objective_history=objective_history,
            n_iter=n_iter,
            converged=converged,
            support_indices=support,
            kkt_violations=violations,
            diagnostics={
                "optimizer": "smo",
                "objective": objective,
                "passes_without_change": passes_without_change,
                "support_count": int(support.size),
            },
        )

    def _choose_second_index(
        self,
        i: int,
        errors: Array,
        alphas: Array,
        C: float,
        rng: np.random.Generator,
        finite_c: bool,
    ) -> int:
        if finite_c:
            non_bound = np.flatnonzero((alphas > self.eps) & (alphas < C - self.eps))
        else:
            non_bound = np.flatnonzero(alphas > self.eps)
        candidates = non_bound[non_bound != i]
        if candidates.size == 0:
            candidates = np.delete(np.arange(errors.size), i)
        if candidates.size == 0:
            return i
        return int(candidates[np.argmax(np.abs(errors[candidates] - errors[i]))])

    def _is_free(self, alpha: float, C: float, finite_c: bool) -> bool:
        if alpha <= self.eps:
            return False
        if finite_c and alpha >= C - self.eps:
            return False
        return True


class ChunkingOptimizer:
    """Approximate the chunking scheme described in Section 2.1 of the paper.

    It repeatedly trains on an active set, keeps current support vectors, and
    adds examples that violate the margin. This is mainly useful for studying
    the original training strategy; SMO is the default optimizer.
    """

    def __init__(
        self,
        base_optimizer: SMOOptimizer | DualQPSolver | None = None,
        chunk_size: int = 256,
        max_rounds: int = 20,
        tol: float = 1e-3,
    ) -> None:
        self.base_optimizer = base_optimizer or SMOOptimizer(tol=tol)
        self.chunk_size = chunk_size
        self.max_rounds = max_rounds
        self.tol = tol

    def solve(
        self,
        x: Array,
        y: Array,
        kernel: KernelCallable,
        C: float | None = 1.0,
    ) -> SVMOptimizationResult:
        x = np.asarray(x, dtype=float)
        y = validate_binary_labels(y)
        n_samples = x.shape[0]
        active = np.arange(min(self.chunk_size, n_samples))
        history: list[float] = []
        result: SVMOptimizationResult | None = None
        result_active = active.copy()

        for round_index in range(self.max_rounds):
            result_active = active.copy()
            result = self.base_optimizer.solve(x[active], y[active], kernel, C)
            history.extend(result.objective_history)
            support_global = active[result.support_indices]
            support_x = x[support_global]
            support_y = y[support_global]
            support_alpha = result.alphas[result.support_indices]
            scores = kernel(x, support_x) @ (support_alpha * support_y) + result.bias
            violators = np.flatnonzero(y * scores < 1.0 - self.tol)
            new_active = np.unique(np.concatenate([support_global, violators]))
            if new_active.size == active.size and np.array_equal(new_active, active):
                break
            if new_active.size == active.size:
                break
            if new_active.size > active.size + self.chunk_size:
                missing = np.setdiff1d(new_active, support_global, assume_unique=False)
                new_active = np.unique(
                    np.concatenate([support_global, missing[: self.chunk_size]])
                )
            active = new_active

        if result is None:
            raise RuntimeError("chunking optimizer failed to run")
        full_alphas = np.zeros(n_samples, dtype=float)
        full_alphas[result_active] = result.alphas
        gram = kernel(x, x)
        bias = compute_bias(full_alphas, y, gram, C)
        support = np.flatnonzero(full_alphas > 1e-6)
        return SVMOptimizationResult(
            alphas=full_alphas,
            bias=bias,
            objective_history=history,
            n_iter=result.n_iter,
            converged=result.converged,
            support_indices=support,
            kkt_violations=kkt_violation_summary(full_alphas, y, gram, bias, C),
            diagnostics={
                "optimizer": "chunking",
                "active_set_size": int(active.size),
                "rounds": round_index + 1,
            },
        )
