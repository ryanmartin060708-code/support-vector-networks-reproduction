"""MNIST/NIST reproduction using the paper's degree-4 polynomial setup."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import markdown_table, precision_recall_f1
from src.svm import OneVsRestSVM, SoftMarginSVM
from src.utils import ensure_dir, load_mnist, subsample_per_class, write_json
from src.visualization import plot_benchmark_errors, plot_digit_grid


ORIGINAL_TABLE_3 = {
    "support_vectors": [1379, 989, 1958, 1900, 1224, 2024, 1527, 2064, 2332, 2765],
    "train_errors": [7, 16, 8, 11, 2, 4, 8, 16, 4, 1],
    "test_errors": [19, 14, 35, 35, 36, 49, 32, 43, 48, 63],
    "combined_test_error_percent": 1.1,
}


def run(args: argparse.Namespace) -> dict[str, object]:
    data_dir = ensure_dir(ROOT / "data" / "mnist")
    results_dir = ensure_dir(ROOT / "results")
    figures_dir = ensure_dir(ROOT / "figures")
    x_train, y_train, x_test, y_test = load_mnist(data_dir, download=args.download)
    x_train, y_train = subsample_per_class(
        x_train, y_train, args.max_train_per_class, seed=args.seed
    )
    x_test, y_test = subsample_per_class(
        x_test, y_test, args.max_test_per_class, seed=args.seed + 1
    )

    base = SoftMarginSVM(
        C=args.C,
        kernel="polynomial",
        degree=4,
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
    train_metrics = precision_recall_f1(y_train, train_pred)
    test_metrics = precision_recall_f1(y_test, test_pred)
    support_counts = clf.support_counts()

    payload = {
        "dataset": "MNIST used as NIST substitute",
        "degree": 4,
        "C": args.C,
        "gamma": args.gamma,
        "train_size": int(y_train.size),
        "test_size": int(y_test.size),
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["macro_precision"],
        "test_recall_macro": test_metrics["macro_recall"],
        "test_f1_macro": test_metrics["macro_f1"],
        "support_vectors_by_class": support_counts.tolist(),
        "support_vectors_total": int(np.sum(support_counts)),
        "support_vectors_mean": float(np.mean(support_counts)),
        "original_table_3": ORIGINAL_TABLE_3,
    }
    save_results(payload, results_dir)

    scores = clf.decision_function(x_test)
    class_labels = clf.classes_
    if 1 in set(class_labels.tolist()):
        class_index = int(np.flatnonzero(class_labels == 1)[0])
        binary_target = np.where(y_test == 1, 1.0, -1.0)
        binary_pred = np.where(scores[:, class_index] >= 0.0, 1.0, -1.0)
        errors = np.flatnonzero(binary_target != binary_pred)
        if errors.size:
            plot_digit_grid(
                x_test[errors[:14]],
                y_test[errors[:14]],
                figures_dir / "figure8_mnist_classifier1_errors.png",
                image_shape=(28, 28),
                max_images=min(14, errors.size),
                title="Figure 8 recreation: class-1 test errors",
            )

    plot_benchmark_errors(
        ["Original SVN", "Reproduced"],
        [
            ORIGINAL_TABLE_3["combined_test_error_percent"],
            100.0 * (1.0 - float(test_metrics["accuracy"])),
        ],
        figures_dir / "mnist_nist_error_comparison.png",
        title="Degree-4 polynomial SVM: original NIST vs reproduced MNIST",
    )
    return payload


def save_results(payload: dict[str, object], results_dir: Path) -> None:
    write_json(results_dir / "mnist_results.json", payload)
    with (results_dir / "mnist_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in payload.items():
            if key == "original_table_3":
                continue
            writer.writerow([key, value])

    rows = [
        ["Original NIST SVN", 1.1, "", "", "", sum(ORIGINAL_TABLE_3["support_vectors"])],
        [
            "Reproduced MNIST SVN",
            100.0 * (1.0 - float(payload["test_accuracy"])),
            payload["test_precision_macro"],
            payload["test_recall_macro"],
            payload["test_f1_macro"],
            payload["support_vectors_total"],
        ],
    ]
    markdown = "# MNIST/NIST Reproduction Results\n\n" + markdown_table(
        ["Experiment", "Error %", "Macro precision", "Macro recall", "Macro F1", "SV total"],
        rows,
    )
    (results_dir / "mnist_results.md").write_text(markdown + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.01,
        help="Polynomial dot-product scale. Use 1.0 for the literal paper kernel.",
    )
    parser.add_argument("--optimizer", choices=["smo", "qp", "chunking"], default="smo")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--max-passes", type=int, default=10)
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    parser.add_argument("--no-store-kernel", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    print(
        f"test_accuracy={payload['test_accuracy']:.4f}, "
        f"macro_f1={payload['test_f1_macro']:.4f}, "
        f"support_vectors={payload['support_vectors_total']}"
    )


if __name__ == "__main__":
    main()
