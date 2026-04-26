import os
import json
import numpy as np

# ==============================================================================
#  GradCAM_to_text.py
#
#  Converts all 800 per-image GradCAM .npy heatmap arrays into
#  natural language clinical text descriptions.
#
#  Input:
#    GradCAM_per_image folder containing .npy files named:
#    fold{N}_{cls}_{img_name}.npy
#    e.g. fold1_CH_image_1776.npy
#
#  Output:
#    gradcam_descriptions.json — dictionary keyed by image name:
#    {
#      "fold1_CH_image_1776": {
#        "text":        "GradCAM shows strong focal activation...",
#        "quadrant":    "superior left",
#        "pattern":     "focal",
#        "strength":    "strong",
#        "peak_x":      198,
#        "peak_y":      142,
#        "high_pct":    2.8,
#        "medium_pct":  14.7,
#        "near_disc":   True,
#        "peripheral":  False,
#        "pred_class":  "CH",
#        "fold":        1
#      },
#      ...
#    }
#
#  Run after gradcam.py in generate_all mode:
#    python GradCAM_to_text.py
# ==============================================================================

# ── Config ────────────────────────────────────────────────────────────────────
PER_IMAGE_DIR = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k\GradCAM_per_image"
OUTPUT_JSON   = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k\gradcam_descriptions.json"

CLASSES       = ["CH", "CO", "Normal", "RB", "RCH", "UM"]
# ─────────────────────────────────────────────────────────────────────────────


# ==============================================================================
#  Core conversion function
# ==============================================================================

def describe_activation(cam, pred_class):
    """
    Convert a raw GradCAM numpy array (H x W, values 0-1)
    into a natural language clinical description.

    Parameters
    ----------
    cam        : np.ndarray (H, W) — raw output from gradcam.generate()
                 values between 0 and 1
    pred_class : str — predicted class e.g. "CH"

    Returns
    -------
    text : str  — clinical description of attention pattern
    meta : dict — structured data for downstream processing
    """
    h, w   = cam.shape
    mid_y  = h // 2
    mid_x  = w // 2

    # ── Mask out black border (UWF circular mask) ──────────────────────────
    border_mask  = cam > 0.01
    valid_pixels = int(border_mask.sum())
    if valid_pixels == 0:
        return "GradCAM: insufficient activation data.", {}

    # ── Find peak activation location ─────────────────────────────────────
    peak_y, peak_x = np.unravel_index(cam.argmax(), cam.shape)

    # ── Determine anatomical quadrant ──────────────────────────────────────
    vertical   = "superior" if peak_y < mid_y else "inferior"
    horizontal = "left"     if peak_x < mid_x else "right"
    quadrant   = f"{vertical} {horizontal}"

    # ── Measure activation coverage ────────────────────────────────────────
    high_pixels   = int((cam > 0.6).sum())
    medium_pixels = int((cam > 0.3).sum())
    high_pct      = high_pixels   / valid_pixels * 100
    medium_pct    = medium_pixels / valid_pixels * 100

    # ── Classify activation pattern ────────────────────────────────────────
    if high_pct < 3.0:
        pattern = "focal"       # tight concentrated spot
    elif high_pct < 12.0:
        pattern = "regional"    # moderate spread
    else:
        pattern = "diffuse"     # widespread activation

    # ── Measure peak intensity ─────────────────────────────────────────────
    peak_val = float(cam.max())
    if peak_val > 0.8:
        strength = "strong"
    elif peak_val > 0.5:
        strength = "moderate"
    else:
        strength = "weak"

    # ── Check proximity to optic disc zone ─────────────────────────────────
    # Optic disc typically in nasal half, vertically centred in UWF images
    disc_roi  = cam[int(h * 0.3):int(h * 0.7), int(w * 0.05):int(w * 0.45)]
    near_disc = bool(disc_roi.max() > 0.5)

    # ── Check peripheral involvement ───────────────────────────────────────
    margin        = 0.15
    top_edge      = float(cam[:int(h * margin), :].max())
    bot_edge      = float(cam[int(h * (1 - margin)):, :].max())
    left_edge     = float(cam[:, :int(w * margin)].max())
    right_edge    = float(cam[:, int(w * (1 - margin)):].max())
    is_peripheral = max(top_edge, bot_edge, left_edge, right_edge) > 0.5

    # ── Build natural language description ─────────────────────────────────
    disc_note = (
        " Activation overlaps with the peripapillary region "
        "adjacent to the optic disc."
        if near_disc else ""
    )
    periph_note = (
        " Peripheral retinal involvement noted."
        if is_peripheral and not near_disc else ""
    )

    text = (
        f"GradCAM shows {strength} {pattern} activation "
        f"in the {quadrant} region of the fundus "
        f"(peak at pixel {int(peak_x)}, {int(peak_y)} "
        f"of {w}x{h}). "
        f"High-activation area covers {high_pct:.1f}% "
        f"of the fundus with moderate activation "
        f"across {medium_pct:.1f}%."
        f"{disc_note}"
        f"{periph_note}"
        f" Pattern consistent with {pred_class} localisation."
    )

    meta = {
        "quadrant":    quadrant,
        "pattern":     pattern,
        "strength":    strength,
        "peak_x":      int(peak_x),
        "peak_y":      int(peak_y),
        "high_pct":    round(float(high_pct), 1),
        "medium_pct":  round(float(medium_pct), 1),
        "near_disc":   bool(near_disc),
        "peripheral":  bool(is_peripheral),
    }

    return text, meta


# ==============================================================================
#  Parse filename to extract fold, class and image name
# ==============================================================================

def parse_filename(fname):
    """
    Parse fold{N}_{cls}_{img_name}.npy into components.

    Examples
    --------
    fold1_CH_image_1776.npy  →  fold=1, cls='CH', img='image_1776'
    fold3_Normal_image_0012.npy  →  fold=3, cls='Normal', img='image_0012'
    """
    name = fname.replace('.npy', '')

    # Extract fold number
    parts = name.split('_', 1)   # ['fold1', 'CH_image_1776']
    if not parts[0].startswith('fold'):
        return None, None, None

    try:
        fold = int(parts[0].replace('fold', ''))
    except ValueError:
        return None, None, None

    rest = parts[1]   # 'CH_image_1776'

    # Extract class — find which class name is at the start
    pred_class = None
    img_name   = None
    for cls in sorted(CLASSES, key=len, reverse=True):  # longest first
        if rest.startswith(cls + '_'):
            pred_class = cls
            img_name   = rest[len(cls) + 1:]   # 'image_1776'
            break

    return fold, pred_class, img_name


# ==============================================================================
#  Main
# ==============================================================================

def main():
    print(f"\n{'='*60}")
    print(f"  GradCAM to Text Converter")
    print(f"  Input : {PER_IMAGE_DIR}")
    print(f"  Output: {OUTPUT_JSON}")
    print(f"{'='*60}\n")

    # Find all .npy files
    if not os.path.exists(PER_IMAGE_DIR):
        print(f"  ⚠️  Folder not found: {PER_IMAGE_DIR}")
        print(f"  Run gradcam.py in generate_all mode first.")
        return

    npy_files = sorted([
        f for f in os.listdir(PER_IMAGE_DIR)
        if f.endswith('.npy')
    ])

    if not npy_files:
        print(f"  ⚠️  No .npy files found in {PER_IMAGE_DIR}")
        print(f"  Run gradcam.py in generate_all mode first.")
        return

    print(f"  Found {len(npy_files)} .npy files\n")

    descriptions = {}
    skipped      = 0
    processed    = 0

    for i, fname in enumerate(npy_files):
        # Parse filename
        fold, pred_class, img_name = parse_filename(fname)

        if pred_class is None:
            print(f"  ⚠️  Could not parse filename: {fname} — skipping")
            skipped += 1
            continue

        # Load heatmap
        npy_path = os.path.join(PER_IMAGE_DIR, fname)
        try:
            cam = np.load(npy_path).astype(np.float32)
        except Exception as e:
            print(f"  ⚠️  Failed to load {fname}: {e}")
            skipped += 1
            continue

        # Convert to text
        text, meta = describe_activation(cam, pred_class)

        if not meta:
            print(f"  ⚠️  Empty activation in {fname} — skipping")
            skipped += 1
            continue

        # Build record
        key = fname.replace('.npy', '')
        descriptions[key] = {
            "text":       text,
            "pred_class": pred_class,
            "fold":       fold,
            "img_name":   img_name,
            **meta
        }
        processed += 1

        # Progress every 100 files
        if (i + 1) % 100 == 0 or (i + 1) == len(npy_files):
            print(f"  [{i+1:4d}/{len(npy_files)}] {fname}")
            print(f"          → {text[:80]}...")

    # Save JSON
    print(f"\n  Saving descriptions to:\n  {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

    print(f"\n  {'='*50}")
    print(f"  Total processed : {processed}")
    print(f"  Skipped         : {skipped}")
    print(f"  JSON saved      : {OUTPUT_JSON}")
    print(f"  {'='*50}")

    # Print sample output per class
    print(f"\n  Sample descriptions per class:")
    seen = set()
    for key, rec in descriptions.items():
        cls = rec['pred_class']
        if cls not in seen:
            print(f"\n  [{cls}] {key}")
            print(f"  {rec['text']}")
            seen.add(cls)
        if len(seen) == len(CLASSES):
            break

    print(f"\n  Next step: integrate gradcam_descriptions.json")
    print(f"  into Report_Generator.py to replace the hardcoded")
    print(f"  GRADCAM_REGIONS dictionary.")
    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()