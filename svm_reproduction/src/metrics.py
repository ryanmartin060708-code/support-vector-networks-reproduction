"""Small metric utilities without depending on scikit-learn."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def accuracy_score(y_true: Array, y_pred: Array) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: Array,
    y_pred: Array,
    labels: Array | None = None,
) -> Array:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {label: i for i, label in enumerate(labels.tolist())}
    matrix = np.zeros((labels.size, labels.size), dtype=int)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[index[actual], index[predicted]] += 1
    return matrix


def precision_recall_f1(
    y_true: Array,
    y_pred: Array,
    labels: Array | None = None,
) -> dict[str, object]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    cm = confusion_matrix(y_true, y_pred, labels)

    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    for idx, label in enumerate(labels):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        per_class[label.item() if hasattr(label, "item") else label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(cm[idx, :])),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a compact GitHub-flavored markdown table."""

    def cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])
