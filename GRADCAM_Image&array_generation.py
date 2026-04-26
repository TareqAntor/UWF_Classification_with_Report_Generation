import matplotlib
matplotlib.use('Agg')

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from PIL import Image

# ==============================================================================
#  gradcam.py  — UPDATED
#
#  Two modes controlled by RUN_MODE:
#
#  MODE 1: "visualise"  (original behaviour)
#    - Processes N_IMAGES_VIS samples per class
#    - Saves per-class PNG figures + combined figure
#    - Same output as before
#
#  MODE 2: "generate_all"  (new)
#    - Processes EVERY test image in all 6 classes
#    - Saves per-image .npy heatmap arrays  → used for GradCAM-to-text
#    - Saves per-image .jpg overlay images  → used for visualisation
#    - Prints progress every 50 images
#
#  Set RUN_MODE below then run:
#    python gradcam.py
# ==============================================================================

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR     = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k"
DATA_DIR        = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Data\Test"


FOLDS           = [1, 2, 3, 4, 5]

# Visualise mode uses fold 1 only for the paper figure
VIS_FOLD        = 1
SAVE_DIR        = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k\GradCAM"

# Per-image output folder (MODE 2)
PER_IMAGE_DIR   = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k\GradCAM_per_image"

CLASSES         = ["CH", "CO", "Normal", "RB", "RCH", "UM"]
IMG_SIZE        = (224, 224)
INPUT_MEAN      = [0.485, 0.456, 0.406]
INPUT_STD       = [0.229, 0.224, 0.225]

# ── Mode selector ─────────────────────────────────────────────────────────────
# "visualise"    → same as original (6 class PNGs + combined figure, fold 1 only)
# "generate_all" → per-image .npy + .jpg for ALL folds (~800 images total)
RUN_MODE        = "generate_all"

N_IMAGES_VIS    = 3    # only used in "visualise" mode
SAVE_JPG        = True  # save coloured overlay .jpg alongside .npy
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PER_IMAGE_DIR, exist_ok=True)


# ==============================================================================
#  GradCAM engine — unchanged from original
# ==============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if output.min() < 0:
            probs = torch.exp(output)
        else:
            probs = output

        pred_idx  = probs.argmax(dim=1).item()
        pred_prob = probs[0, pred_idx].item()

        if class_idx is None:
            class_idx = pred_idx

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients   = self.gradients
        activations = self.activations

        if activations.dim() == 3:
            activations = activations[:, 1:, :]
            gradients   = gradients[:, 1:, :]

            n_patches = activations.shape[1]
            grid_size = int(n_patches ** 0.5)

            weights = gradients.mean(dim=2)
            cam     = (weights.unsqueeze(-1) * activations).sum(-1)
            cam     = cam.reshape(1, grid_size, grid_size)
            cam     = cam[0].cpu().numpy()

        else:
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam     = (weights * activations).sum(dim=1)
            cam     = cam[0].cpu().numpy()

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, IMG_SIZE)
        return cam, pred_idx, pred_prob


# ==============================================================================
#  Target layer finder — unchanged from original
# ==============================================================================

def get_target_layer(model):
    m = model.module if hasattr(model, 'module') else model

    if hasattr(m, 'blocks'):
        target = m.blocks[-1].norm1
        print(f"  Target layer: ViT blocks[-1].norm1")
        return target

    if hasattr(m, 'stages'):
        target = m.stages[-1].blocks[-1]
        print(f"  Target layer: ConvNeXt stages[-1].blocks[-1]")
        return target

    if hasattr(m, 'layer4'):
        target = m.layer4[-1]
        print(f"  Target layer: ResNet layer4[-1]")
        return target

    if hasattr(m, 'features'):
        target = m.features[-1]
        print(f"  Target layer: DenseNet features[-1]")
        return target

    last_layer = None
    for name, module in m.named_modules():
        if 'head' not in name and 'classifier' not in name and 'fc' not in name:
            last_layer = module
    print(f"  Target layer: fallback last non-classifier layer")
    return last_layer


# ==============================================================================
#  Image utilities — unchanged from original
# ==============================================================================

def preprocess(img_path):
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=INPUT_MEAN, std=INPUT_STD),
    ])
    img_pil  = Image.open(img_path).convert('RGB')
    tensor   = tf(img_pil).unsqueeze(0)
    img_orig = np.array(img_pil.resize(IMG_SIZE))
    return tensor, img_orig


def apply_heatmap(img_orig, cam):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay       = cv2.addWeighted(img_orig, 0.5, heatmap_color, 0.5, 0)
    return overlay, heatmap_color


# ==============================================================================
#  Plotting — unchanged from original
# ==============================================================================

def plot_gradcam_row(images_data, class_name, save_path):
    n   = len(images_data)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original', 'GradCAM heatmap', 'Overlay']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=10)

    for row, (img_orig, heatmap, overlay, pred_cls, pred_prob) in enumerate(images_data):
        axes[row, 0].imshow(img_orig)
        axes[row, 0].set_ylabel(f'Sample {row+1}', fontsize=10, rotation=90, labelpad=8)
        axes[row, 1].imshow(heatmap)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_xlabel(
            f'Pred: {pred_cls}  ({pred_prob*100:.1f}%)',
            fontsize=9, color='green' if pred_cls == class_name else 'red'
        )
        for ax in axes[row]:
            ax.axis('off')

    fig.suptitle(f'GradCAM — Class: {class_name}', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_combined_figure(all_class_data, categories, save_path):
    n_cls = len(categories)
    fig, axes = plt.subplots(n_cls, 2, figsize=(7, 3.5 * n_cls))

    for row, cls in enumerate(categories):
        if cls not in all_class_data or not all_class_data[cls]:
            continue
        img_orig, heatmap, overlay, pred_cls, pred_prob = all_class_data[cls][0]

        axes[row, 0].imshow(img_orig)
        axes[row, 0].set_ylabel(cls, fontsize=12, fontweight='bold', rotation=0,
                                 labelpad=40, va='center')
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_xlabel(
            f'Pred: {pred_cls} ({pred_prob*100:.1f}%)',
            fontsize=9, color='darkgreen' if pred_cls == cls else 'darkred'
        )
        for ax in [axes[row, 0], axes[row, 1]]:
            ax.axis('off')

    axes[0, 0].set_title('Original',        fontsize=12, fontweight='bold')
    axes[0, 1].set_title('GradCAM overlay', fontsize=12, fontweight='bold')
    fig.suptitle('GradCAM Visualizations — All Classes',
                 fontsize=14, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Combined figure saved: {save_path}")


# ==============================================================================
#  MODE 1 — Visualise (original behaviour, 3 samples per class)
# ==============================================================================

def run_visualise(model_dict, categories, device):
    """Visualise mode — fold 1 only, same as original behaviour."""
    fold        = VIS_FOLD
    ckpt_path   = os.path.join(RESULTS_DIR, f"vit_base_patch16_224_in21k_fold_{fold}.pt")
    ckpt        = torch.load(ckpt_path, weights_only=False)
    model       = ckpt['model'].to(device)
    model.eval()
    idx_to_class = ckpt.get('idx_to_class', {i: c for i, c in enumerate(categories)})

    target_layer = get_target_layer(model)
    gradcam      = GradCAM(model, target_layer)

    test_dir = os.path.join(DATA_DIR, f"fold_{fold}")
    print(f"\n  Mode: VISUALISE ({N_IMAGES_VIS} images per class, fold {fold})")
    all_class_data = {}

    for cls in categories:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"  ⚠️  Folder not found: {cls_dir}")
            continue

        img_files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:N_IMAGES_VIS]

        if not img_files:
            continue

        print(f"\n  Processing class: {cls} ({len(img_files)} images)")
        class_results = []

        for img_file in img_files:
            img_path         = os.path.join(cls_dir, img_file)
            tensor, img_orig = preprocess(img_path)
            tensor           = tensor.to(device)

            cam, pred_idx, pred_prob = gradcam.generate(tensor)
            pred_cls = idx_to_class.get(pred_idx, str(pred_idx))

            overlay, heatmap_color = apply_heatmap(img_orig, cam)
            class_results.append((img_orig, heatmap_color, overlay, pred_cls, pred_prob))

            correct = "CORRECT" if pred_cls == cls else "WRONG"
            print(f"    {img_file}  →  pred: {pred_cls} ({pred_prob*100:.1f}%)  [{correct}]")

        save_path = os.path.join(SAVE_DIR, f"GradCAM_{cls}.png")
        plot_gradcam_row(class_results, cls, save_path)
        all_class_data[cls] = class_results

    combined_path = os.path.join(SAVE_DIR, "GradCAM_ALL_CLASSES.png")
    plot_combined_figure(all_class_data, categories, combined_path)


# ==============================================================================
#  MODE 2 — Generate all per-image heatmaps (.npy + .jpg)
# ==============================================================================

def run_generate_all(categories, device):
    """
    Process every image across all 5 folds.
    For each image saves:
      - fold{N}_{cls}_{img_name}.npy  — raw heatmap array (H x W, float32, 0-1)
      - fold{N}_{cls}_{img_name}.jpg  — coloured overlay (if SAVE_JPG=True)

    Total output: ~800 .npy files (one per test image across all folds).
    The .npy files are used by GradCAM_to_text.py to generate
    image-specific text descriptions for report generation.
    """
    print(f"\n  Mode: GENERATE ALL — processing {len(FOLDS)} folds")
    print(f"  Folds: {FOLDS}")
    print(f"  Output: {PER_IMAGE_DIR}")
    print(f"  Save JPG overlays: {SAVE_JPG}\n")

    grand_total  = 0
    grand_saved  = 0

    for fold in FOLDS:
        # Load fold-specific checkpoint
        ckpt_path = os.path.join(
            RESULTS_DIR,
            f"vit_base_patch16_224_in21k_fold_{fold}.pt"
        )
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  Checkpoint not found for fold {fold}: {ckpt_path}")
            continue

        print(f"\n  {'─'*50}")
        print(f"  Loading fold {fold}: {os.path.basename(ckpt_path)}")
        ckpt         = torch.load(ckpt_path, weights_only=False)
        model        = ckpt['model'].to(device)
        model.eval()
        idx_to_class = ckpt.get('idx_to_class', {i: c for i, c in enumerate(categories)})

        # Re-initialise GradCAM for this fold's model
        target_layer = get_target_layer(model)
        gradcam      = GradCAM(model, target_layer)

        test_dir = os.path.join(DATA_DIR, f"fold_{fold}")

        fold_total = 0
        fold_saved = 0

        for cls in categories:
            cls_dir = os.path.join(test_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"  ⚠️  Folder not found: {cls_dir}")
                continue

            img_files = sorted([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])

            print(f"\n  Fold {fold} | Class: {cls} — {len(img_files)} images")

            for i, img_file in enumerate(img_files):
                img_path         = os.path.join(cls_dir, img_file)
                tensor, img_orig = preprocess(img_path)
                tensor           = tensor.to(device)

                # Generate heatmap
                cam, pred_idx, pred_prob = gradcam.generate(tensor)
                pred_cls = idx_to_class.get(pred_idx, str(pred_idx))

                # Safe filename — fold + class + image name
                base_name = (img_file.replace('.jpg', '')
                                     .replace('.jpeg', '')
                                     .replace('.png', ''))
                safe_name = f"fold{fold}_{cls}_{base_name}"

                # Save raw numpy array — key output for GradCAM-to-text
                npy_path = os.path.join(PER_IMAGE_DIR, f"{safe_name}.npy")
                np.save(npy_path, cam.astype(np.float32))
                fold_saved += 1

                # Optionally save coloured overlay JPG
                if SAVE_JPG:
                    overlay, _ = apply_heatmap(img_orig, cam)
                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    jpg_path    = os.path.join(PER_IMAGE_DIR, f"{safe_name}.jpg")
                    cv2.imwrite(jpg_path, overlay_bgr)

                fold_total += 1

                # Progress every 50 images
                if (i + 1) % 50 == 0 or (i + 1) == len(img_files):
                    correct = "✓" if pred_cls == cls else "✗"
                    print(f"    [{i+1:3d}/{len(img_files)}] {img_file} "
                          f"→ {pred_cls} ({pred_prob*100:.1f}%) {correct}")

        print(f"\n  Fold {fold} complete — {fold_total} images, {fold_saved} .npy files saved")
        grand_total += fold_total
        grand_saved += fold_saved

    print(f"\n  {'='*50}")
    print(f"  ALL FOLDS COMPLETE")
    print(f"  Total images processed : {grand_total}")
    print(f"  Total .npy files saved : {grand_saved}")
    print(f"  Output folder          : {PER_IMAGE_DIR}")
    print(f"  {'='*50}")
    print(f"\n  Next step: run GradCAM_to_text.py to convert")
    print(f"  the .npy files into clinical text descriptions.")


# ==============================================================================
#  Main
# ==============================================================================

def main():
    print(f"\n{'='*60}")
    print(f"  GradCAM Generator  —  Mode: {RUN_MODE.upper()}")
    print(f"{'='*60}\n")

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    categories = CLASSES
    print(f"  Device  : {device}")
    print(f"  Classes : {categories}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PER_IMAGE_DIR, exist_ok=True)

    if RUN_MODE == "visualise":
        run_visualise(None, categories, device)

    elif RUN_MODE == "generate_all":
        run_generate_all(categories, device)

    else:
        print(f"  ⚠️  Unknown RUN_MODE: '{RUN_MODE}'")
        print(f"  Set RUN_MODE to 'visualise' or 'generate_all'")

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()