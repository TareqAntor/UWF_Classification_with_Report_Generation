import os
import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

# ==============================================================================
#  generate_results.py
#  Prints per-class Sensitivity, Specificity, Precision, F1
#  as mean +- std across all 5 folds.
#
#  Run after training:
#    python generate_results.py
# ==============================================================================

# ── Config — update for each model ────────────────────────────────────────────
RESULTS_PATH = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k"
MODEL_NAME   = "vit_base_patch16_224_in21k"
NUM_FOLDS    = 5
CLASSES      = ["CH", "CO", "Normal", "RB", "RCH", "UM"]  # fallback
# ─────────────────────────────────────────────────────────────────────────────


def compute_per_class(all_targets, pred_label, categories):
    cm_pc     = multilabel_confusion_matrix(all_targets, pred_label)
    per_class = {}
    for i, cls in enumerate(categories):
        tn = cm_pc[i][0][0];  fp = cm_pc[i][0][1]
        fn = cm_pc[i][1][0];  tp = cm_pc[i][1][1]
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        f1c  = 2 * prec * sens / (prec + sens + 1e-8)
        per_class[cls] = {
            'sensitivity': round(sens * 100, 2),
            'specificity': round(spec * 100, 2),
            'precision':   round(prec * 100, 2),
            'f1':          round(f1c  * 100, 2),
        }
    return per_class


def main():
    all_fold_per_class = []
    categories         = CLASSES

    for fold_idx in range(1, NUM_FOLDS + 1):
        ckpt_path = os.path.join(
            RESULTS_PATH,
            f"{MODEL_NAME}_test_fold_{fold_idx}.pt"
        )
        if not os.path.exists(ckpt_path):
            print(f"  Warning: not found — {ckpt_path}")
            continue
        ckpt        = torch.load(ckpt_path, weights_only=False)
        all_targets = ckpt['targets']
        pred_label  = ckpt['prediction_label']
        categories  = ckpt.get('categories', CLASSES)
        per_class   = compute_per_class(all_targets, pred_label, categories)
        all_fold_per_class.append(per_class)

    if not all_fold_per_class:
        print("No checkpoints found. Check RESULTS_PATH and MODEL_NAME.")
        return

    w = 72
    print('\n' + '-' * w)
    print('  PER-CLASS BREAKDOWN  --  mean +- std across all folds')
    print('-' * w)
    print(
        f"  {'Class':<10}"
        f"  {'Sens% (mean+-std)':>18}"
        f"  {'Spec% (mean+-std)':>18}"
        f"  {'Prec% (mean+-std)':>18}"
        f"  {'F1% (mean+-std)':>16}"
    )
    print('  ' + '-' * (w - 2))

    for cls in categories:
        sens_vals = [f[cls]['sensitivity'] for f in all_fold_per_class]
        spec_vals = [f[cls]['specificity'] for f in all_fold_per_class]
        prec_vals = [f[cls]['precision']   for f in all_fold_per_class]
        f1_vals   = [f[cls]['f1']          for f in all_fold_per_class]
        print(
            f"  {cls:<10}"
            f"  {np.mean(sens_vals):>6.2f} +- {np.std(sens_vals):<5.2f}"
            f"  {np.mean(spec_vals):>6.2f} +- {np.std(spec_vals):<5.2f}"
            f"  {np.mean(prec_vals):>6.2f} +- {np.std(prec_vals):<5.2f}"
            f"  {np.mean(f1_vals):>6.2f} +- {np.std(f1_vals):<5.2f}"
        )

    print('-' * w)
    print(f"  Computed from {len(all_fold_per_class)} folds.\n")


if __name__ == "__main__":
    main()