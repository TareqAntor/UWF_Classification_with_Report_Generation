import os
import json

# ================================================================
#  CONFIG — only change these lines
# ================================================================
GRADCAM_JSON = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k\gradcam_descriptions.json"
REPORTS_DIR  = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k\Reports"
# ================================================================


TEMPLATE = """\
================================================================================
  CLINICAL DIAGNOSTIC REPORT — AI GENERATED
  UWF Fundus Intraocular Tumor Classification System
================================================================================

Image         : {img_name}.jpg
Date          : April 2026

--------------------------------------------------------------------------------
1. PATIENT INFORMATION
--------------------------------------------------------------------------------
Patient ID    : {img_name}
Imaging date  : April 2026
Modality      : Ultra-wide-field (UWF) fundus photography
Predicted     : {pred_class}
Fold          : {fold}

--------------------------------------------------------------------------------
2. CLINICAL INTERPRETATION
--------------------------------------------------------------------------------
{gradcam_text}

--------------------------------------------------------------------------------
3. KEY VISUAL FEATURES (GradCAM Analysis)
--------------------------------------------------------------------------------
Attention pattern  : {pattern} ({strength})
Quadrant           : {quadrant}
Peak coordinates   : ({peak_x}, {peak_y}) of 224x224
High activation    : {high_pct}% of fundus
Moderate activation: {medium_pct}% of fundus
Disc proximity     : {disc_note}
Peripheral         : {periph_note}

================================================================================
  END OF REPORT
  For research use only — NOT for clinical deployment
================================================================================
"""


def main():
    print(f"\n{'='*60}")
    print(f"  GradCAM Report Generator")
    print(f"  Input : {GRADCAM_JSON}")
    print(f"  Output: {REPORTS_DIR}")
    print(f"{'='*60}\n")

    # Load JSON
    if not os.path.exists(GRADCAM_JSON):
        print(f"  ⚠️  JSON not found: {GRADCAM_JSON}")
        print(f"  Run GradCAM_to_text.py first.")
        return

    with open(GRADCAM_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} GradCAM entries\n")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    generated = 0
    skipped   = 0

    for key, rec in data.items():

        img_name    = rec.get('img_name',   key)
        pred_class  = rec.get('pred_class', 'Unknown')
        fold        = rec.get('fold',       '?')
        gradcam_text = rec.get('text',      'No GradCAM description available.')
        pattern     = rec.get('pattern',    'unknown')
        strength    = rec.get('strength',   'unknown')
        quadrant    = rec.get('quadrant',   'unknown')
        peak_x      = rec.get('peak_x',     0)
        peak_y      = rec.get('peak_y',     0)
        high_pct    = rec.get('high_pct',   0.0)
        medium_pct  = rec.get('medium_pct', 0.0)
        near_disc   = rec.get('near_disc',  False)
        peripheral  = rec.get('peripheral', False)

        disc_note   = "Yes — activation overlaps peripapillary region" if near_disc  else "No"
        periph_note = "Yes — peripheral retinal involvement noted"     if peripheral else "No"

        report = TEMPLATE.format(
            img_name    = img_name,
            pred_class  = pred_class,
            fold        = fold,
            gradcam_text = gradcam_text,
            pattern     = pattern,
            strength    = strength,
            quadrant    = quadrant,
            peak_x      = peak_x,
            peak_y      = peak_y,
            high_pct    = high_pct,
            medium_pct  = medium_pct,
            disc_note   = disc_note,
            periph_note = periph_note,
        )

        out_path = os.path.join(REPORTS_DIR, f"report_{img_name}.txt")
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(report)
            generated += 1
        except Exception as e:
            print(f"  ⚠️  Failed to save {img_name}: {e}")
            skipped += 1
            continue

        if generated % 200 == 0:
            print(f"  [{generated}/{len(data)}] {img_name} → {pred_class}")

    print(f"\n  {'='*50}")
    print(f"  Total generated : {generated}")
    print(f"  Skipped         : {skipped}")
    print(f"  Saved to        : {REPORTS_DIR}")
    print(f"  {'='*50}\n")


if __name__ == "__main__":
    main()