import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================================================================
#  plot_confusion_matrix.py
#
#  Plots:
#    1. Per-fold confusion matrix  (one plot per fold)
#    2. Cumulative confusion matrix (all folds combined)
#    3. Normalized confusion matrix (shows % instead of counts)
#
#  Usage:
#    python plot_confusion_matrix.py
# ==============================================================================

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_PATH = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k"
MODEL_NAME   = "vit_base_patch16_224_in21k"
NUM_FOLDS    = 5
SAVE_FIGS    = True   # True = save as PNG,  False = just display

# Class names in the order your model uses
# (will be auto-loaded from checkpoint, but defined here as fallback)
CLASSES = ["CH", "CO", "Normal", "RB", "RCH", "UM"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def plot_cm(cm, classes, title, save_path=None, normalize=False, cmap="Blues"):
    """
    Plot a single confusion matrix.

    Parameters
    ----------
    cm         : np.ndarray  raw confusion matrix (counts)
    classes    : list[str]   class names
    title      : str         plot title
    save_path  : str|None    if given, saves figure to this path
    normalize  : bool        if True, show row-normalized percentages
    cmap       : str         matplotlib colormap
    """
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt     = ".2f"
        vmax    = 1.0
    else:
        cm_plot = cm
        fmt     = "d"
        vmax    = cm.max()

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11)

    # Annotate cells
    thresh = cm_plot.max() / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = f"{cm_plot[i, j]:{fmt}}" if not normalize \
                  else f"{cm_plot[i, j]*100:.1f}%"
            ax.text(j, i, val,
                    ha="center", va="center", fontsize=10,
                    color="white" if cm_plot[i, j] > thresh else "black")

    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(title,              fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  💾 Saved: {save_path}")

    plt.show()
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    cumulative_cm = None
    fold_cms      = []

    print(f"\n{'='*55}")
    print(f"  Confusion Matrix Report — {MODEL_NAME}")
    print(f"{'='*55}")

    for fold_idx in range(1, NUM_FOLDS + 1):

        ckpt_path = os.path.join(
            RESULTS_PATH,
            f"{MODEL_NAME}_test_fold_{fold_idx}.pt"
        )

        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  Checkpoint not found: {ckpt_path}")
            continue

        # Load checkpoint
        ckpt    = torch.load(ckpt_path, weights_only=False)
        targets = ckpt["targets"]
        preds   = ckpt["prediction_label"]

        # Use class names from checkpoint if available
        classes = ckpt.get("categories", CLASSES)

        # Compute confusion matrix for this fold
        cm = confusion_matrix(targets, preds, labels=list(range(len(classes))))
        fold_cms.append(cm)

        # Accumulate for cumulative CM
        if cumulative_cm is None:
            cumulative_cm = cm.copy()
        else:
            cumulative_cm += cm

        # Per-fold accuracy from CM diagonal
        fold_acc = np.sum(np.diagonal(cm)) / np.sum(cm) * 100
        print(f"\n  Fold {fold_idx}  —  Accuracy: {fold_acc:.2f}%")
        print(f"  Confusion Matrix:")
        print(f"  {'':10}", end="")
        for cls in classes:
            print(f"  {cls:>6}", end="")
        print()
        for i, cls in enumerate(classes):
            print(f"  {cls:>10}", end="")
            for j in range(len(classes)):
                print(f"  {cm[i,j]:>6}", end="")
            print()

        # ── Plot: raw counts ─────────────────────────────────────────────────
        save_path_raw  = os.path.join(
            RESULTS_PATH,
            f"CM_fold_{fold_idx}_counts.png"
        ) if SAVE_FIGS else None

        plot_cm(
            cm=cm,
            classes=classes,
            title=f"Confusion Matrix — Fold {fold_idx}  (counts)\n{MODEL_NAME}",
            save_path=save_path_raw,
            normalize=False,
        )

        # ── Plot: normalized (%) ─────────────────────────────────────────────
        save_path_norm = os.path.join(
            RESULTS_PATH,
            f"CM_fold_{fold_idx}_normalized.png"
        ) if SAVE_FIGS else None

        plot_cm(
            cm=cm,
            classes=classes,
            title=f"Confusion Matrix — Fold {fold_idx}  (normalized %)\n{MODEL_NAME}",
            save_path=save_path_norm,
            normalize=True,
            cmap="Greens",
        )

    # ── Cumulative CM (all folds combined) ────────────────────────────────────
    if cumulative_cm is not None:

        overall_acc = np.sum(np.diagonal(cumulative_cm)) / np.sum(cumulative_cm) * 100
        print(f"\n{'='*55}")
        print(f"  CUMULATIVE (all {NUM_FOLDS} folds combined)")
        print(f"  Overall Accuracy: {overall_acc:.2f}%")
        print(f"{'='*55}")

        # Raw cumulative
        plot_cm(
            cm=cumulative_cm,
            classes=classes,
            title=f"Cumulative Confusion Matrix — All {NUM_FOLDS} Folds (counts)\n{MODEL_NAME}",
            save_path=os.path.join(RESULTS_PATH, "CM_cumulative_counts.png") if SAVE_FIGS else None,
            normalize=False,
        )

        # Normalized cumulative
        plot_cm(
            cm=cumulative_cm,
            classes=classes,
            title=f"Cumulative Confusion Matrix — All {NUM_FOLDS} Folds (normalized %)\n{MODEL_NAME}",
            save_path=os.path.join(RESULTS_PATH, "CM_cumulative_normalized.png") if SAVE_FIGS else None,
            normalize=True,
            cmap="Greens",
        )

        # ── Mean ± Std CM across folds ────────────────────────────────────────
        if len(fold_cms) == NUM_FOLDS:
            stack    = np.stack(fold_cms, axis=0)   # shape (5, 6, 6)
            mean_cm  = stack.mean(axis=0)
            std_cm   = stack.std(axis=0)

            print(f"\n  Mean ± Std Confusion Matrix (across {NUM_FOLDS} folds):")
            print(f"  {'':10}", end="")
            for cls in classes:
                print(f"  {cls:>12}", end="")
            print()
            for i, cls in enumerate(classes):
                print(f"  {cls:>10}", end="")
                for j in range(len(classes)):
                    print(f"  {mean_cm[i,j]:>5.1f}±{std_cm[i,j]:>4.1f}", end="")
                print()

    print(f"\n{'='*55}")
    print(f"  Done! Figures saved to:\n  {RESULTS_PATH}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()