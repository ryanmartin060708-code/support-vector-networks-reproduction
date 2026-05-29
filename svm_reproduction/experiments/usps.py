"""USPS digit experiment from Cortes and Vapnik, Table 2."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kernels import polynomial_feature_dimension
from src.metrics import accuracy_score, markdown_table
from src.svm import OneVsRestSVM, SoftMarginSVM
from src.utils import (
    ensure_dir,
    load_usps,
    preprocess_usps,
    subsample_per_class,
    write_json,
)
from src.visualization import plot_digit_grid


ORIGINAL_TABLE_2 = {
    1: {"raw_error_percent": 12.0, "support_vectors_mean": 200, "feature_space": "256"},
    2: {"raw_error_percent": 4.7, "support_vectors_mean": 127, "feature_space": "~33000"},
    3: {"raw_error_percent": 4.4, "support_vectors_mean": 148, "feature_space": "~1e6"},
    4: {"raw_error_percent": 4.3, "support_vectors_mean": 165, "feature_space": "~1e9"},
    5: {"raw_error_percent": 4.3, "support_vectors_mean": 175, "feature_space": "~1e12"},
    6: {"raw_error_percent": 4.2, "support_vectors_mean": 185, "feature_space": "~1e14"},
    7: {"raw_error_percent": 4.3, "support_vectors_mean": 190, "feature_space": "~1e16"},
}


def run(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    data_dir = ensure_dir(ROOT / "data" / "usps")
    results_dir = ensure_dir(ROOT / "results")
    figures_dir = ensure_dir(ROOT / "figures")
    x_train, y_train, x_test, y_test = load_usps(data_dir, download=args.download)

    if args.preprocess:
        x_train = preprocess_usps(x_train, smooth_sigma=args.smooth_sigma)
        x_test = preprocess_usps(x_test, smooth_sigma=args.smooth_sigma)
    else:
        x_train = np.asarray(x_train, dtype=float)
        x_test = np.asarray(x_test, dtype=float)

    x_train, y_train = subsample_per_class(
        x_train, y_train, args.max_train_per_class, seed=args.seed
    )
    x_test, y_test = subsample_per_class(
        x_test, y_test, args.max_test_per_class, seed=args.seed + 1
    )

    rows: list[dict[str, float | int | str]] = []
    degree_two_model: OneVsRestSVM | None = None
    for degree in args.degrees:
        base = SoftMarginSVM(
            C=args.C,
            kernel="polynomial",
            degree=degree,
            coef0=1.0,
            gamma=args.gamma,
            optimizer=args.optimizer,
            tol=args.tol,
            eps=args.eps,
            max_iter=args.max_iter,
            max_passes=args.max_passes,
            random_state=args.seed,
            store_kernel=not args.no_store_kernel,
        )
        clf = OneVsRestSVM(base)
        clf.fit(x_train, y_train)
        train_pred = clf.predict(x_train)
        test_pred = clf.predict(x_test)
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        support_counts = clf.support_counts()
        original = ORIGINAL_TABLE_2.get(degree, {})
        rows.append(
            {
                "degree": degree,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "raw_error_percent": 100.0 * (1.0 - test_accuracy),
                "support_vectors_mean": float(np.mean(support_counts)),
                "support_vectors_total": int(np.sum(support_counts)),
                "original_raw_error_percent": original.get("raw_error_percent", ""),
                "original_support_vectors_mean": original.get("support_vectors_mean", ""),
                "feature_space_dimension": polynomial_feature_dimension(
                    x_train.shape[1], degree, include_bias=False
                ),
            }
        )
        if degree == 2:
            degree_two_model = clf

    save_results(rows, results_dir)
    if degree_two_model is not None:
        pred = degree_two_model.predict(x_train)
        errors = np.flatnonzero(pred != y_train)
        if errors.size:
            plot_digit_grid(
                x_train[errors[:16]],
                y_train[errors[:16]],
                figures_dir / "figure7_usps_training_errors_degree2.png",
                image_shape=(16, 16),
                max_images=min(16, errors.size),
                title="Figure 7 recreation: USPS degree-2 training errors",
            )
    return rows


def save_results(rows: list[dict[str, float | int | str]], results_dir: Path) -> None:
    csv_path = results_dir / "usps_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(results_dir / "usps_results.json", rows)

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["degree"],
                row["original_raw_error_percent"],
                row["raw_error_percent"],
                row["train_accuracy"],
                row["test_accuracy"],
                row["original_support_vectors_mean"],
                row["support_vectors_mean"],
            ]
        )
    markdown = "# USPS Reproduction Results\n\n" + markdown_table(
        [
            "Degree",
            "Original raw error %",
            "Reproduced raw error %",
            "Train accuracy",
            "Test accuracy",
            "Original mean SV",
            "Reproduced mean SV",
        ],
        table_rows,
    )
    (results_dir / "usps_results.md").write_text(markdown + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["smo", "qp", "chunking"], default="smo")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--max-passes", type=int, default=10)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    parser.add_argument("--preprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-sigma", type=float, default=0.75)
    parser.add_argument("--no-store-kernel", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run(args)
    for row in rows:
        print(
            f"degree={row['degree']}: test_accuracy={row['test_accuracy']:.4f}, "
            f"mean_sv={row['support_vectors_mean']:.1f}"
        )


if __name__ == "__main__":
    main()
