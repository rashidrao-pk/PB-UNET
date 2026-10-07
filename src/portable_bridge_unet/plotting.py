"""Headless plots for saved training history and per-image evaluation metrics."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORES = ("dice", "iou", "precision", "recall", "accuracy")

def _save(fig, base):
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        for extension in ("png", "pdf"):
            fig.savefig(base.with_suffix("." + extension), dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)

def plot_training_history(history, out_dir):
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    for ax, metric in zip(axes.flat, ("loss", *SCORES)):
        for prefix, label in (("train", "Training"), ("val", "Validation")):
            key = f"{prefix}_{metric}"
            if any(key in row for row in history):
                ax.plot(epochs, [row.get(key, np.nan) for row in history], label=label, marker=".", markersize=3)
        ax.set(title=metric.upper() if metric == "iou" else metric.capitalize(), xlabel="Epoch", ylabel=metric.capitalize())
        if metric != "loss":
            ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend()
    ax = axes.flat[6]
    ax.plot(epochs, [row.get("lr", np.nan) for row in history], color="tab:green")
    ax.set(title="Learning rate", xlabel="Epoch", ylabel="Learning rate", yscale="log")
    ax.grid(alpha=0.25)
    axes.flat[7].set_visible(False)
    fig.suptitle("Training and validation history")
    _save(fig, Path(out_dir) / "training_curves")

def plot_evaluation_metrics(frame, out_dir, suffix="raw"):
    """Use all finite scores; report excluded nonfinite values on each chart."""
    if frame.empty:
        return
    columns = [key for key in (*SCORES, "hd95") if key in frame]
    if not columns:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, key in zip(axes.flat, columns):
        values = frame[key].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        if valid.size:
            ax.hist(valid, bins=np.linspace(0, 1, 21) if key != "hd95" else min(20, max(1, len(valid))), color="tab:blue", edgecolor="white")
            ax.axvline(valid.mean(), color="tab:orange", linestyle="--", label=f"Mean {valid.mean():.3f}")
            ax.legend()
        ax.set(title=f"{key.upper()} (n={len(valid)}, nonfinite={len(values)-len(valid)})",
               xlabel="Distance (pixels)" if key == "hd95" else "Score", ylabel="Images")
        ax.grid(axis="y", alpha=0.25)
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    fig.suptitle(f"Per-image evaluation distributions — {suffix}")
    _save(fig, Path(out_dir) / f"metric_distributions_{suffix}")

    scores = [key for key in SCORES if key in frame and np.isfinite(frame[key].to_numpy(dtype=float)).any()]
    if scores:
        arrays = [frame[key].to_numpy(dtype=float) for key in scores]
        arrays = [values[np.isfinite(values)] for values in arrays]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        axes[0].bar(range(len(scores)), [a.mean() for a in arrays],
                    yerr=[a.std(ddof=1) if len(a)>1 else 0 for a in arrays], capsize=4)
        axes[0].set(title="Mean score with sample standard deviation", ylabel="Score")
        axes[1].boxplot(arrays)
        axes[1].set(title="Per-image score distributions", ylabel="Score")
        for ax in axes:
            ax.set_xticks(range(len(scores)) if ax is axes[0] else range(1, len(scores)+1), [key.upper() for key in scores])
            ax.set_ylim(-0.05, 1.08)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Evaluation summary — {suffix}")
        _save(fig, Path(out_dir) / f"metric_summary_{suffix}")
