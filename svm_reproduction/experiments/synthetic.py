"""Reproduce the paper's 2D polynomial SVM illustrations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import accuracy_score
from src.svm import HardMarginSVM, SoftMarginSVM
from src.utils import ensure_dir, write_json
from src.visualization import (
    plot_benchmark_errors,
    plot_decision_boundary,
    plot_network_schematic,
)


def make_quadratic_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, 2))
    boundary = 0.55 * x[:, 0] ** 2 - 0.55
    y = np.where(x[:, 1] > boundary, 1.0, -1.0)
    return x, y


def make_xor(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.95, size=(n, 2))
    y = np.where(x[:, 0] * x[:, 1] >= 0.0, 1.0, -1.0)
    flip = rng.random(n) < 0.03
    y[flip] *= -1.0
    return x, y


def make_ellipse(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.1, 2.1, size=(n, 2))
    radius = (x[:, 0] / 1.25) ** 2 + (x[:, 1] / 0.85) ** 2
    y = np.where(radius <= 1.0, 1.0, -1.0)
    flip = rng.random(n) < 0.02
    y[flip] *= -1.0
    return x, y


def make_overlapping(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    x_pos = rng.normal([0.7, 0.55], [0.7, 0.55], size=(n1, 2))
    x_neg = rng.normal([-0.65, -0.35], [0.75, 0.65], size=(n2, 2))
    x = np.vstack([x_pos, x_neg])
    y = np.r_[np.ones(n1), -np.ones(n2)]
    order = rng.permutation(n)
    return x[order], y[order]


DATASETS = {
    "quadratic": make_quadratic_split,
    "xor": make_xor,
    "ellipse": make_ellipse,
    "overlap": make_overlapping,
}


def run(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    figures_dir = ensure_dir(ROOT / "figures" / "synthetic")
    results_dir = ensure_dir(ROOT / "results")
    summary: dict[str, dict[str, float]] = {}

    for offset, (name, maker) in enumerate(DATASETS.items()):
        x, y = maker(args.n_samples, args.seed + offset)
        model = SoftMarginSVM(
            C=args.C,
            kernel="polynomial",
            degree=2,
            coef0=1.0,
            optimizer=args.optimizer,
            tol=args.tol,
            max_iter=args.max_iter,
            max_passes=args.max_passes,
            random_state=args.seed + offset,
        )
        model.fit(x, y)
        pred = model.predict(x)
        accuracy = accuracy_score(y, pred)
        diagnostics = model.get_diagnostics()
        summary[name] = {
            "training_accuracy": accuracy,
            "training_error_percent": 100.0 * (1.0 - accuracy),
            "support_vectors": float(diagnostics["support_vectors"]),
            "dual_objective": float(diagnostics["dual_objective"]),
            "kkt_max": float(diagnostics["kkt"]["max"]),
        }
        plot_decision_boundary(
            model,
            x,
            y,
            figures_dir / f"{name}_degree2.png",
            title=f"{name}: degree-2 polynomial SVM",
        )

    recreate_paper_figures(args)
    write_json(results_dir / "synthetic_results.json", summary)
    return summary


def recreate_paper_figures(args: argparse.Namespace) -> None:
    """Generate approximations of Figures 2, 3, 4, 5, and 9."""

    figures_dir = ensure_dir(ROOT / "figures")
    x, y = make_overlapping(70, args.seed + 100)
    model = SoftMarginSVM(
        C=args.C,
        kernel="linear",
        optimizer=args.optimizer,
        tol=args.tol,
        max_iter=args.max_iter,
        max_passes=args.max_passes,
        random_state=args.seed + 100,
    )
    model.fit(x, y)
    plot_decision_boundary(
        model,
        x,
        y,
        figures_dir / "figure2_margin_example.png",
        title="Figure 2 recreation: maximum-margin separator",
    )
    plot_network_schematic(figures_dir / "figure3_feature_space_network.png", kind="feature")
    plot_network_schematic(figures_dir / "figure4_kernel_network.png", kind="kernel")
    plot_benchmark_errors(
        ["linear", "k=3 NN", "LeNet1", "LeNet4", "SVN"],
        [12.0, 2.4, 1.7, 1.1, 1.1],
        figures_dir / "figure9_benchmark_recreation.png",
        title="Figure 9 recreation: NIST benchmark errors",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=90)
    parser.add_argument("--C", type=float, default=20.0)
    parser.add_argument("--optimizer", choices=["smo", "qp", "chunking"], default="smo")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--max-passes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    for name, values in summary.items():
        print(
            f"{name}: accuracy={values['training_accuracy']:.3f}, "
            f"support_vectors={int(values['support_vectors'])}"
        )


if __name__ == "__main__":
    main()
