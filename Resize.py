import os
import cv2
from pathlib import Path

# ==============================================================================
#  resize_dataset.py
#  Resizes all images in your 5-fold dataset structure.
#
#  Source structure (your current 512×512 images):
#    Data/Train/fold_1/CH, CO, Normal, RB, RCH, UM
#    Data/Train/fold_2/ ...
#    Data/Val/fold_1/  ...
#    Data/Test/fold_1/ ...
#
#  Output structure (resized images saved to a NEW folder — originals untouched):
#    Data_resized/Train/fold_1/CH, CO, Normal, RB, RCH, UM
#    Data_resized/Train/fold_2/ ...
#    ...
# ==============================================================================

# ── Config ────────────────────────────────────────────────────────────────────

SOURCE_ROOT = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Data"
OUTPUT_ROOT = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Data_384"

TARGET_SIZE = (384, 384)   # ← change to (224, 224) or (456, 456) as needed

SPLITS      = ["Train", "Val", "Test"]
NUM_FOLDS   = 5
CLASSES     = ["CH", "CO", "Normal", "RB", "RCH", "UM"]
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ── Resize ────────────────────────────────────────────────────────────────────

def resize_dataset(source_root, output_root, target_size, splits, num_folds, classes):

    total_processed = 0
    total_skipped   = 0

    for split in splits:
        for fold_idx in range(1, num_folds + 1):
            fold_name = f"fold_{fold_idx}"

            for cls in classes:
                src_dir = os.path.join(source_root, split, fold_name, cls)
                dst_dir = os.path.join(output_root, split, fold_name, cls)

                if not os.path.exists(src_dir):
                    print(f"  ⚠️  Missing: {src_dir}")
                    continue

                os.makedirs(dst_dir, exist_ok=True)

                images = [
                    f for f in os.listdir(src_dir)
                    if Path(f).suffix.lower() in IMG_EXTS
                ]

                for img_name in images:
                    src_path = os.path.join(src_dir, img_name)
                    dst_path = os.path.join(dst_dir, img_name)

                    # Skip if already resized
                    if os.path.exists(dst_path):
                        total_skipped += 1
                        continue

                    img = cv2.imread(src_path)

                    if img is None:
                        print(f"  ⚠️  Could not read: {src_path}")
                        continue

                    resized = cv2.resize(
                        img, target_size,
                        interpolation=cv2.INTER_CUBIC   # best quality downscale
                    )

                    cv2.imwrite(dst_path, resized)
                    total_processed += 1

                print(
                    f"  ✅ {split}/fold_{fold_idx}/{cls}: "
                    f"{len(images)} images → {target_size[0]}×{target_size[1]}"
                )

    print(f"\n{'='*55}")
    print(f"  Done!  Processed: {total_processed}  |  Skipped: {total_skipped}")
    print(f"  Resized dataset saved to:\n  {output_root}")
    print(f"{'='*55}")


# ── Verify sizes (optional sanity check) ─────────────────────────────────────

def verify_sizes(output_root, target_size, splits, num_folds, classes, sample_n=3):
    """Spot-check a few images per class to confirm correct resize."""
    print(f"\n{'='*55}")
    print("  VERIFICATION — spot-checking resized images")
    print(f"{'='*55}")
    all_ok = True

    for split in splits[:1]:   # check Train only
        for fold_idx in [1]:    # check fold_1 only
            for cls in classes:
                cls_dir = os.path.join(output_root, split, f"fold_{fold_idx}", cls)
                if not os.path.exists(cls_dir):
                    continue
                images = [f for f in os.listdir(cls_dir)
                          if Path(f).suffix.lower() in IMG_EXTS][:sample_n]
                for img_name in images:
                    img = cv2.imread(os.path.join(cls_dir, img_name))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    ok = (h == target_size[1] and w == target_size[0])
                    status = "✅" if ok else "❌"
                    if not ok:
                        all_ok = False
                    print(f"  {status}  {split}/fold_1/{cls}/{img_name}  →  {w}×{h}")

    print(f"\n  {'All sizes correct ✅' if all_ok else 'Size mismatch found ❌ — check output'}")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*55}")
    print(f"  Resizing dataset: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"  Source : {SOURCE_ROOT}")
    print(f"  Output : {OUTPUT_ROOT}")
    print(f"{'='*55}\n")

    resize_dataset(
        source_root=SOURCE_ROOT,
        output_root=OUTPUT_ROOT,
        target_size=TARGET_SIZE,
        splits=SPLITS,
        num_folds=NUM_FOLDS,
        classes=CLASSES,
    )

    verify_sizes(
        output_root=OUTPUT_ROOT,
        target_size=TARGET_SIZE,
        splits=SPLITS,
        num_folds=NUM_FOLDS,
        classes=CLASSES,
    )