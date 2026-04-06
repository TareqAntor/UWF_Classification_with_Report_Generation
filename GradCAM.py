import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from PIL import Image

# ==============================================================================
#  gradcam.py
#
#  Generates GradCAM heatmap overlays for your trained ViT/ConvNeXt models.
#
#  Output:
#    One figure per class (6 total), each showing:
#      - Original image
#      - GradCAM heatmap
#      - Overlay (heatmap on image)
#    Plus one combined figure with all 6 classes side by side.
#
#  Run after training:
#    python gradcam.py
# ==============================================================================

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k\vit_base_patch16_224_in21k_fold_1.pt"
TEST_DATA_DIR   = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Data\Test\fold_1"
SAVE_DIR        = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k\GradCAM"

CLASSES     = ["CH", "CO", "Normal", "RB", "RCH", "UM"]
IMG_SIZE    = (224, 224)       # must match what you trained with
INPUT_MEAN  = [0.485, 0.456, 0.406]
INPUT_STD   = [0.229, 0.224, 0.225]
N_IMAGES    = 3                # number of sample images per class
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)


# ==============================================================================
#  GradCAM engine
# ==============================================================================

class GradCAM:
    """
    Works with any model that has convolutional or attention layers.
    Automatically finds the best target layer for ViT and ConvNeXt.
    """

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
        """
        Generate GradCAM heatmap for a single image tensor.

        Parameters
        ----------
        input_tensor : torch.Tensor  (1, C, H, W)
        class_idx    : int|None      target class; if None uses predicted class

        Returns
        -------
        heatmap : np.ndarray  (H, W)  values in [0, 1]
        pred_idx: int         predicted class index
        pred_prob: float      predicted probability
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # Handle LogSoftmax output (your CNN_Classifier uses LogSoftmax)
        if output.min() < 0:
            probs = torch.exp(output)
        else:
            probs = output

        pred_idx  = probs.argmax(dim=1).item()
        pred_prob = probs[0, pred_idx].item()

        if class_idx is None:
            class_idx = pred_idx

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # GradCAM computation
        gradients   = self.gradients   # (1, C, H, W) or (1, N, C) for ViT
        activations = self.activations # (1, C, H, W) or (1, N, C) for ViT

        # Handle ViT: activations are (1, num_patches+1, embed_dim)
        if activations.dim() == 3:
            # Remove CLS token, reshape patches to 2D grid
            activations = activations[:, 1:, :]   # (1, N, D)
            gradients   = gradients[:, 1:, :]      # (1, N, D)

            n_patches = activations.shape[1]
            grid_size = int(n_patches ** 0.5)

            # Weight activations by gradients
            weights     = gradients.mean(dim=2)                          # (1, N)
            cam         = (weights.unsqueeze(-1) * activations).sum(-1)  # (1, N)
            cam         = cam.reshape(1, grid_size, grid_size)            # (1, g, g)
            cam         = cam[0].cpu().numpy()

        else:
            # CNN: activations are (1, C, H, W)
            weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
            cam     = (weights * activations).sum(dim=1)          # (1, H, W)
            cam     = cam[0].cpu().numpy()

        # ReLU + normalize
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = cv2.resize(cam, IMG_SIZE)
        return cam, pred_idx, pred_prob


# ==============================================================================
#  Target layer finder — works for ViT and ConvNeXt
# ==============================================================================

def get_target_layer(model):
    """
    Automatically find the best GradCAM target layer.
    Returns the last meaningful feature layer for both ViT and ConvNeXt.
    """
    model_name = type(model).__name__.lower()

    # Unwrap DataParallel if used
    m = model.module if hasattr(model, 'module') else model

    # ── ViT (timm) ───────────────────────────────────────────────────────────
    if hasattr(m, 'blocks'):
        # timm ViT: model.blocks[-1].norm1
        target = m.blocks[-1].norm1
        print(f"  Target layer: ViT blocks[-1].norm1")
        return target

    # ── ConvNeXt (timm) ──────────────────────────────────────────────────────
    if hasattr(m, 'stages'):
        # timm ConvNeXt: model.stages[-1].blocks[-1]
        target = m.stages[-1].blocks[-1]
        print(f"  Target layer: ConvNeXt stages[-1].blocks[-1]")
        return target

    # ── ResNet (torchvision) ─────────────────────────────────────────────────
    if hasattr(m, 'layer4'):
        target = m.layer4[-1]
        print(f"  Target layer: ResNet layer4[-1]")
        return target

    # ── DenseNet ─────────────────────────────────────────────────────────────
    if hasattr(m, 'features'):
        target = m.features[-1]
        print(f"  Target layer: DenseNet features[-1]")
        return target

    # ── Fallback: last named module that is not a classifier ─────────────────
    last_layer = None
    for name, module in m.named_modules():
        if 'head' not in name and 'classifier' not in name and 'fc' not in name:
            last_layer = module
    print(f"  Target layer: fallback last non-classifier layer")
    return last_layer


# ==============================================================================
#  Image utilities
# ==============================================================================

def preprocess(img_path):
    """Load and preprocess a single image for model input."""
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=INPUT_MEAN, std=INPUT_STD),
    ])
    img_pil  = Image.open(img_path).convert('RGB')
    tensor   = tf(img_pil).unsqueeze(0)   # (1, 3, H, W)
    img_orig = np.array(img_pil.resize(IMG_SIZE))
    return tensor, img_orig


def apply_heatmap(img_orig, cam):
    """
    Overlay GradCAM heatmap on original image.

    Parameters
    ----------
    img_orig : np.ndarray  (H, W, 3)  uint8 RGB
    cam      : np.ndarray  (H, W)     float [0, 1]

    Returns
    -------
    overlay : np.ndarray  (H, W, 3)  uint8 RGB
    heatmap : np.ndarray  (H, W, 3)  uint8 RGB  (colormap only)
    """
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay       = cv2.addWeighted(img_orig, 0.5, heatmap_color, 0.5, 0)
    return overlay, heatmap_color


# ==============================================================================
#  Plotting
# ==============================================================================

def plot_gradcam_row(images_data, class_name, save_path):
    """
    Plot N_IMAGES × 3 grid (original | heatmap | overlay) for one class.

    Parameters
    ----------
    images_data : list of (img_orig, heatmap, overlay, pred_class, pred_prob)
    class_name  : str
    save_path   : str
    """
    n   = len(images_data)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original', 'GradCAM heatmap', 'Overlay']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=10)

    for row, (img_orig, heatmap, overlay, pred_cls, pred_prob) in enumerate(images_data):
        axes[row, 0].imshow(img_orig)
        axes[row, 0].set_ylabel(
            f'Sample {row+1}', fontsize=10, rotation=90, labelpad=8
        )
        axes[row, 1].imshow(heatmap)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_xlabel(
            f'Pred: {pred_cls}  ({pred_prob*100:.1f}%)',
            fontsize=9, color='green' if pred_cls == class_name else 'red'
        )
        for ax in axes[row]:
            ax.axis('off')

    fig.suptitle(
        f'GradCAM — Class: {class_name}',
        fontsize=15, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_combined_figure(all_class_data, categories, save_path):
    """
    Combined figure: one row per class, columns = (original | overlay).
    Paper-ready figure showing all 6 classes together.
    """
    n_cls  = len(categories)
    fig, axes = plt.subplots(n_cls, 2, figsize=(7, 3.5 * n_cls))

    for row, cls in enumerate(categories):
        if cls not in all_class_data or not all_class_data[cls]:
            continue
        img_orig, heatmap, overlay, pred_cls, pred_prob = all_class_data[cls][0]

        axes[row, 0].imshow(img_orig)
        axes[row, 0].set_title(
            f'{cls}' if row == 0 else '', fontsize=11
        )
        axes[row, 0].set_ylabel(cls, fontsize=12, fontweight='bold', rotation=0,
                                 labelpad=40, va='center')
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_xlabel(
            f'Pred: {pred_cls} ({pred_prob*100:.1f}%)',
            fontsize=9,
            color='darkgreen' if pred_cls == cls else 'darkred'
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
#  Main
# ==============================================================================

def main():

    print(f"\n{'='*60}")
    print(f"  GradCAM Generator")
    print(f"  Model:  {os.path.basename(CHECKPOINT_PATH)}")
    print(f"  Output: {SAVE_DIR}")
    print(f"{'='*60}\n")

    # ── Load model ─────────────────────────────────────────────────────────
    print("  Loading model checkpoint...")
    ckpt       = torch.load(CHECKPOINT_PATH, weights_only=False)
    model      = ckpt['model']
    categories = ckpt.get('categories', CLASSES)
    idx_to_class = ckpt.get('idx_to_class', {i: c for i, c in enumerate(categories)})

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = model.to(device)
    model.eval()
    print(f"  Device: {device}")
    print(f"  Classes: {categories}")

    # ── Find target layer ──────────────────────────────────────────────────
    print("\n  Finding GradCAM target layer...")
    target_layer = get_target_layer(model)
    gradcam      = GradCAM(model, target_layer)

    # ── Collect sample images per class ───────────────────────────────────
    all_class_data = {}

    for cls in categories:
        cls_dir = os.path.join(TEST_DATA_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"  ⚠️  Folder not found: {cls_dir}")
            continue

        img_files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:N_IMAGES]

        if not img_files:
            print(f"  ⚠️  No images found in {cls_dir}")
            continue

        print(f"\n  Processing class: {cls} ({len(img_files)} images)")
        class_results = []

        for img_file in img_files:
            img_path = os.path.join(cls_dir, img_file)

            # Preprocess
            tensor, img_orig = preprocess(img_path)
            tensor = tensor.to(device)

            # GradCAM
            cam, pred_idx, pred_prob = gradcam.generate(tensor)
            pred_cls = idx_to_class.get(pred_idx, str(pred_idx))

            # Heatmap overlay
            overlay, heatmap_color = apply_heatmap(img_orig, cam)

            class_results.append((img_orig, heatmap_color, overlay, pred_cls, pred_prob))
            correct = "CORRECT" if pred_cls == cls else "WRONG"
            print(f"    {img_file}  →  pred: {pred_cls} ({pred_prob*100:.1f}%)  [{correct}]")

        # Per-class figure
        save_path = os.path.join(SAVE_DIR, f"GradCAM_{cls}.png")
        plot_gradcam_row(class_results, cls, save_path)
        all_class_data[cls] = class_results

    # ── Combined paper figure ──────────────────────────────────────────────
    combined_path = os.path.join(SAVE_DIR, "GradCAM_ALL_CLASSES.png")
    plot_combined_figure(all_class_data, categories, combined_path)

    print(f"\n{'='*60}")
    print(f"  Done! All figures saved to:\n  {SAVE_DIR}")
    print(f"  Files generated:")
    for cls in categories:
        print(f"    GradCAM_{cls}.png")
    print(f"    GradCAM_ALL_CLASSES.png  ← use this in your paper")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()