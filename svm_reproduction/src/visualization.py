"""Plotting helpers for paper-style figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import ensure_dir

Array = np.ndarray


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for figure generation. Install requirements.txt."
        ) from exc
    return plt


def plot_decision_boundary(
    model,
    x: Array,
    y: Array,
    path: str | Path,
    title: str | None = None,
    grid_size: int = 300,
) -> Path:
    """Plot 2D decision boundary, margins, errors, and support vectors."""

    plt = _plt()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    if x.shape[1] != 2:
        raise ValueError("decision boundary plot requires 2D features")
    path = Path(path)
    ensure_dir(path.parent)

    padding = 0.35
    x_min, x_max = x[:, 0].min() - padding, x[:, 0].max() + padding
    y_min, y_max = x[:, 1].min() - padding, x[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid).reshape(xx.shape)
    predictions = model.predict(x)
    errors = predictions != y

    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=160)
    ax.contourf(xx, yy, scores, levels=30, cmap="RdBu_r", alpha=0.18)
    ax.contour(
        xx,
        yy,
        scores,
        levels=[-1.0, 0.0, 1.0],
        colors=["#555555", "#111111", "#555555"],
        linestyles=["--", "-", "--"],
        linewidths=[1.1, 1.6, 1.1],
    )
    neg = y < 0
    pos = ~neg
    ax.scatter(x[neg, 0], x[neg, 1], c="#222222", s=28, label="-1", zorder=3)
    ax.scatter(
        x[pos, 0],
        x[pos, 1],
        c="#f7f7f7",
        edgecolors="#222222",
        s=34,
        label="+1",
        zorder=3,
    )
    if hasattr(model, "support_vectors_") and model.support_vectors_ is not None:
        sv = model.support_vectors_
        ax.scatter(
            sv[:, 0],
            sv[:, 1],
            facecolors="none",
            edgecolors="#00897b",
            s=92,
            linewidths=1.6,
            label="support vectors",
            zorder=4,
        )
    if np.any(errors):
        ax.scatter(
            x[errors, 0],
            x[errors, 1],
            marker="x",
            c="#d81b60",
            s=60,
            linewidths=1.7,
            label="training errors",
            zorder=5,
        )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_digit_grid(
    images: Array,
    labels: Array | None,
    path: str | Path,
    image_shape: tuple[int, int],
    max_images: int = 25,
    title: str | None = None,
) -> Path:
    plt = _plt()
    path = Path(path)
    ensure_dir(path.parent)
    images = np.asarray(images)[:max_images]
    labels = None if labels is None else np.asarray(labels)[:max_images]
    cols = min(7, max_images)
    rows = int(np.ceil(images.shape[0] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.25), dpi=160)
    axes = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= images.shape[0]:
            continue
        ax.imshow(images[idx].reshape(image_shape), cmap="gray_r", interpolation="nearest")
        if labels is not None:
            ax.set_title(str(labels[idx]), fontsize=8)
    if title:
        fig.suptitle(title, y=0.98, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_benchmark_errors(
    labels: list[str],
    errors: list[float],
    path: str | Path,
    title: str,
    ylabel: str = "Test error (%)",
) -> Path:
    plt = _plt()
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    ax.bar(labels, errors, color=colors[: len(labels)], width=0.66)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(errors):
        ax.text(idx, value + max(errors) * 0.025, f"{value:g}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_network_schematic(path: str | Path, kind: str = "feature") -> Path:
    """Create simple recreations of the paper's Figure 3/Figure 4 schematics."""

    plt = _plt()
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=160)
    ax.axis("off")
    if kind == "feature":
        boxes = [
            (0.05, 0.45, "input x"),
            (0.32, 0.45, "phi(x)"),
            (0.60, 0.45, "dot with support vectors"),
            (0.86, 0.45, "sign"),
        ]
        title = "Feature-space support-vector network"
    else:
        boxes = [
            (0.06, 0.45, "input x"),
            (0.31, 0.45, "K(x, x_i)"),
            (0.58, 0.45, "alpha_i y_i"),
            (0.84, 0.45, "sum + b"),
        ]
        title = "Kernelized support-vector network"
    for x0, y0, text in boxes:
        ax.add_patch(
            plt.Rectangle((x0, y0), 0.18, 0.22, fill=False, linewidth=1.2, color="#222222")
        )
        ax.text(x0 + 0.09, y0 + 0.11, text, ha="center", va="center", fontsize=8)
    for x0, _, _ in boxes[:-1]:
        ax.annotate(
            "",
            xy=(x0 + 0.25, 0.56),
            xytext=(x0 + 0.19, 0.56),
            arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "#222222"},
        )
    ax.text(0.5, 0.85, title, ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
