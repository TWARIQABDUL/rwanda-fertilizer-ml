"""Metrics and plotting helpers shared by the modelling notebooks."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

CLASS_CLR = {"No": "#2a78d6", "Yes": "#eb6834"}
MODEL_CLR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MUTED = "#52514e"


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11,
        "lines.linewidth": 2,
    })


def scores_of(model, X):
    """Continuous score for ranking metrics: probability if available, else decision function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def metrics(y_true, y_score, y_pred):
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


def evaluate(model, X, y, threshold=None):
    """Metrics at the model's default decision rule, or at a probability threshold if given."""
    s = scores_of(model, X)
    pred = (s >= threshold).astype(int) if threshold is not None else model.predict(X)
    return metrics(y, s, pred)


def best_f1_threshold(y_true, prob):
    p, r, t = precision_recall_curve(y_true, prob)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    i = np.nanargmax(f1[:-1])
    return float(t[i]), float(f1[i])


def plot_confusion(y_true, y_pred, title, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax = ax or plt.gca()
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.0%} of row)", ha="center", va="center",
                    color="white" if pct[i, j] > 0.6 else "#0b0b0b", fontsize=10)
    ax.set_xticks([0, 1], ["No", "Yes"]); ax.set_yticks([0, 1], ["No", "Yes"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title); ax.grid(False)
    return cm


def plot_roc_pr(curves, y_true, title_suffix="", fname=None):
    """curves: dict name -> scores. Draws ROC and precision-recall side by side."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for k, (name, s) in enumerate(curves.items()):
        c = MODEL_CLR[k % len(MODEL_CLR)]
        fpr, tpr, _ = roc_curve(y_true, s)
        a.plot(fpr, tpr, color=c, label=f"{name} (AUC {roc_auc_score(y_true, s):.3f})")
        p, r, _ = precision_recall_curve(y_true, s)
        b.plot(r, p, color=c, label=f"{name} (AP {average_precision_score(y_true, s):.3f})")
    a.plot([0, 1], [0, 1], ls="--", lw=1, color="#8a8984")
    b.axhline(np.mean(y_true), ls="--", lw=1, color="#8a8984")
    b.text(0.99, np.mean(y_true) + 0.015, "no-skill baseline", ha="right", fontsize=8, color=MUTED)
    a.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title=f"ROC curve{title_suffix}")
    b.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall curve{title_suffix}", ylim=(0, 1.02))
    a.legend(frameon=False, fontsize=8, loc="lower right"); b.legend(frameon=False, fontsize=8, loc="lower left")
    plt.tight_layout()
    if fname: plt.savefig(fname, bbox_inches="tight")
    plt.show()


def plot_learning_curve(sizes, train_scores, val_scores, title, ax, metric="ROC-AUC", val_label="Cross-validation"):
    for sc, name, c in [(train_scores, "Training", "#2a78d6"), (val_scores, val_label, "#eb6834")]:
        m, s = sc.mean(axis=1), sc.std(axis=1)
        ax.plot(sizes, m, "o-", color=c, label=name, markersize=5)
        ax.fill_between(sizes, m - s, m + s, color=c, alpha=0.15, linewidth=0)
    ax.set(xlabel="Training plots", ylabel=metric, title=title)
    ax.legend(frameon=False, loc="lower right")
