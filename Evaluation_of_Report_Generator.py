import torch
import json
import os
import numpy as np
from datetime import date
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# ================================================================
#  CONFIG
# ================================================================
CKPT_PATH  = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k\vit_base_patch16_224_in21k_test_fold_1.pt"
OUTPUT_DIR = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\Reports"
GENERATE_REPORTS = True    # True = generate reports using Mistral
                           # False = skip generation, only evaluate existing reports
# ================================================================


# ================================================================
#  SECTION 1 — KNOWLEDGE BASE
# ================================================================

CLASS_DESCRIPTIONS = {
    "Normal": (
        "No intraocular tumor detected. "
        "Optic disc and macula appear within normal limits."
    ),
    "CH": (
        "Choroidal Hemangioma (CH) — a benign vascular tumor of the choroid. "
        "Typically reddish-orange, located in the posterior pole near the optic disc. "
        "May cause exudative retinal detachment if untreated."
    ),
    "CO": (
        "Choroidal Osteoma (CO) — a rare benign ossifying tumor of the choroid. "
        "Appears as a well-defined yellow-white lesion in the juxtapapillary region. "
        "May cause visual field defects."
    ),
    "RB": (
        "Retinoblastoma (RB) — a malignant intraocular tumor and the most common "
        "primary ocular malignancy in children. Presents as white tumour masses "
        "(leukocoria). Life-threatening without prompt treatment."
    ),
    "RCH": (
        "Retinal Capillary Hemangioma (RCH) — a benign vascular hamartoma of the "
        "retina. Appears as a small bright red lesion with dilated feeding vessels. "
        "May be associated with Von Hippel-Lindau disease."
    ),
    "UM": (
        "Uveal Melanoma (UM) — the most common primary intraocular malignancy in "
        "adults. Carries high metastatic potential, particularly to the liver. "
        "Prognosis worsens significantly after metastasis develops."
    ),
}

URGENCY = {
    "Normal": "Routine follow-up as clinically indicated.",
    "CH":     "Non-urgent ophthalmic review within 4 weeks.",
    "CO":     "Non-urgent ophthalmic review within 4 weeks.",
    "RB":     "URGENT — same-day ocular oncology referral required.",
    "RCH":    "Semi-urgent ophthalmic review within 2 weeks.",
    "UM":     "URGENT — same-day ocular oncology referral required.",
}

GRADCAM_REGIONS = {
    "Normal": "optic disc edge and macular region — consistent with normal anatomy",
    "CH":     "posterior pole adjacent to optic disc",
    "CO":     "juxtapapillary choroidal region with calcified appearance",
    "RB":     "inferior quadrant focal white tumour mass",
    "RCH":    "peripheral retinal vascular lesion with feeding vessels",
    "UM":     "superior peripheral choroidal region",
}

# Reference reports — one per class
# These are the ground truth reports your generated reports are compared against.
# Written from clinical knowledge of each class.
# Replace with ophthalmologist-written references for publication.
REFERENCE_REPORTS = {
    "Normal": (
        "No intraocular pathology identified. The optic disc appears healthy with "
        "a sharp well-defined margin. The macula and foveal reflex are normal. "
        "Retinal vasculature is unremarkable. No tumour or lesion detected. "
        "Routine follow-up is recommended."
    ),
    "CH": (
        "Findings are consistent with choroidal hemangioma, a benign vascular "
        "tumour of the choroid. The lesion appears reddish-orange in the posterior "
        "pole adjacent to the optic disc. Non-urgent ophthalmic review is recommended. "
        "Monitor for exudative retinal detachment."
    ),
    "CO": (
        "Findings are consistent with choroidal osteoma, a rare benign ossifying "
        "tumour of the choroid. A well-defined yellow-white lesion is noted in the "
        "juxtapapillary region. Non-urgent ophthalmic review is advised. "
        "Visual field assessment is recommended."
    ),
    "RB": (
        "Findings are consistent with retinoblastoma, a malignant intraocular tumour "
        "most common in children. White tumour masses consistent with leukocoria are "
        "identified. This is a life-threatening condition requiring immediate intervention. "
        "Urgent same-day referral to an ocular oncologist is mandatory."
    ),
    "RCH": (
        "Findings are consistent with retinal capillary hemangioma, a benign vascular "
        "hamartoma of the retina. A small bright red lesion with dilated feeding and "
        "draining vessels is identified peripherally. Von Hippel-Lindau disease should "
        "be excluded. Semi-urgent ophthalmic review within two weeks is advised."
    ),
    "UM": (
        "Findings are consistent with uveal melanoma, the most common primary "
        "intraocular malignancy in adults. The lesion is identified in the superior "
        "peripheral choroidal region. High metastatic potential to the liver is noted. "
        "Urgent same-day referral to an ocular oncologist is mandatory."
    ),
}

# Required clinical keywords that must appear in a correct report
REQUIRED_KEYWORDS = {
    "Normal": ["normal", "optic disc", "macula", "no tumor", "routine"],
    "CH":     ["choroidal", "hemangioma", "benign", "vascular", "optic disc"],
    "CO":     ["choroidal", "osteoma", "benign", "juxtapapillary", "ossif"],
    "RB":     ["retinoblastoma", "malignant", "urgent", "children", "leukocoria"],
    "RCH":    ["capillary", "hemangioma", "retinal", "vascular", "Von Hippel"],
    "UM":     ["melanoma", "uveal", "malignant", "urgent", "metastatic"],
}

# Dangerous keywords — hallucinations that would be clinically harmful
DANGER_KEYWORDS = {
    "Normal": ["tumor", "malignant", "urgent", "cancer", "melanoma"],
    "CH":     ["malignant", "urgent", "retinoblastoma", "melanoma"],
    "CO":     ["malignant", "urgent", "retinoblastoma", "melanoma"],
    "RB":     ["benign", "routine", "no tumor", "normal"],
    "RCH":    ["malignant", "urgent", "retinoblastoma", "melanoma"],
    "UM":     ["benign", "routine", "no tumor", "normal"],
}


# ================================================================
#  SECTION 2 — REPORT GENERATION
# ================================================================

def build_prompt(pred_class, confidence, patient_id, eye_side):
    description    = CLASS_DESCRIPTIONS.get(pred_class, "Unknown.")
    urgency        = URGENCY.get(pred_class, "Clinical review recommended.")
    gradcam_region = GRADCAM_REGIONS.get(pred_class, "posterior fundus")

    prompt = f"""You are a senior ophthalmology AI assistant.
Generate a structured clinical diagnostic report based on the AI model output below.
Be professional, concise, and suitable for a referring ophthalmologist.
Do not invent any patient history. Base the report only on the information provided.

=== AI MODEL OUTPUT ===
Patient ID       : {patient_id}
Eye              : {eye_side}
AI Model         : ViT-B IN-21k (Accuracy: 97.64%, AUC: 99.82%, 5-fold CV)
Predicted class  : {pred_class}
Model confidence : {confidence:.1f}%
GradCAM focus    : {gradcam_region}
Clinical note    : {description}
Urgency          : {urgency}

=== GENERATE A REPORT WITH EXACTLY THESE 6 SECTIONS ===
1. Patient Information
2. AI Classification Result
3. Clinical Interpretation
4. GradCAM Explainability
5. Recommended Action
6. Disclaimer

Keep the total report under 300 words. Use clear clinical language."""

    return prompt


def load_mistral():
    model_id  = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto")
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer,
                   max_new_tokens=512, temperature=0.3,
                   do_sample=True, repetition_penalty=1.1)
    print("Mistral-7B loaded.")
    return gen


def generate_report(generator, prompt):
    messages = [{"role": "user", "content": prompt}]
    output   = generator(messages)
    return output[0]["generated_text"][-1]["content"]


# ================================================================
#  SECTION 3 — EVALUATION METRICS
# ================================================================

def compute_bleu(reference, generated):
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()
    smoothie   = SmoothingFunction().method1
    score      = sentence_bleu([ref_tokens], gen_tokens,
                                smoothing_function=smoothie)
    return round(score, 4)


def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        'rouge1': round(scores['rouge1'].fmeasure, 4),
        'rouge2': round(scores['rouge2'].fmeasure, 4),
        'rougeL': round(scores['rougeL'].fmeasure, 4),
    }


def compute_bertscore(reference, generated):
    P, R, F1 = bert_score_fn(
        [generated], [reference], lang="en", verbose=False)
    return round(float(F1.mean()), 4)


def check_clinical_accuracy(report_text, pred_class):
    """Check what % of required clinical keywords are present."""
    report_lower = report_text.lower()
    required     = REQUIRED_KEYWORDS.get(pred_class, [])
    if not required:
        return 100.0, [], []
    found   = [kw for kw in required if kw.lower() in report_lower]
    missing = [kw for kw in required if kw.lower() not in report_lower]
    score   = len(found) / len(required) * 100
    return round(score, 1), found, missing


def check_hallucinations(report_text, pred_class):
    """Check if any dangerous/incorrect clinical keywords are present."""
    report_lower = report_text.lower()
    dangers      = DANGER_KEYWORDS.get(pred_class, [])
    found_danger = [kw for kw in dangers if kw.lower() in report_lower]
    is_safe      = len(found_danger) == 0
    return is_safe, found_danger


def evaluate_report(report_text, pred_class, true_class):
    """Run all automatic metrics on one report."""
    reference = REFERENCE_REPORTS.get(pred_class, "")

    bleu      = compute_bleu(reference, report_text)
    rouge     = compute_rouge(reference, report_text)
    bert_f1   = compute_bertscore(reference, report_text)
    clin_acc, found_kw, missing_kw = check_clinical_accuracy(report_text, pred_class)
    is_safe, danger_kw             = check_hallucinations(report_text, pred_class)

    return {
        'bleu':          bleu,
        'rouge1':        rouge['rouge1'],
        'rouge2':        rouge['rouge2'],
        'rougeL':        rouge['rougeL'],
        'bert_f1':       bert_f1,
        'clinical_acc':  clin_acc,
        'is_safe':       is_safe,
        'found_keywords':   found_kw,
        'missing_keywords': missing_kw,
        'danger_keywords':  danger_kw,
        'correct_pred':  (pred_class == true_class),
    }


# ================================================================
#  SECTION 4 — MAIN EVALUATION LOOP
# ================================================================

def run_evaluation():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load checkpoint
    print("Loading checkpoint ...")
    ckpt         = torch.load(CKPT_PATH, weights_only=False)
    idx_to_class = ckpt['idx_to_class']
    n_images     = len(ckpt['targets'])
    print(f"Total test images: {n_images}")

    # Load LLM if generating reports
    generator = None
    if GENERATE_REPORTS:
        generator = load_mistral()

    # Storage for results
    all_results  = []
    class_scores = {cls: [] for cls in idx_to_class.values()}

    print(f"\n{'='*65}")
    print(f"{'Idx':<5} {'Image':<30} {'True':<8} {'Pred':<8} {'BLEU':<6} {'ROUGE-L':<8} {'BERT':<6} {'ClinAcc':<8} {'Safe'}")
    print(f"{'='*65}")

    for i in range(n_images):

        # Extract prediction
        pred_idx   = int(ckpt['prediction_label'][i])
        true_idx   = int(ckpt['targets'][i])
        pred_class = idx_to_class[pred_idx]
        true_class = idx_to_class[true_idx]
        confidence = float(ckpt['prediction_probs'][i][pred_idx]) * 100
        img_name   = str(ckpt['image_names'][i]).split("\\")[-1]

        patient_id = f"P-{i:05d}"
        eye_side   = "Right eye"

        # Generate or load report
        safe_name  = img_name.replace(".jpg","").replace("/","_")
        report_path = os.path.join(OUTPUT_DIR, f"report_{safe_name}.txt")

        if GENERATE_REPORTS:
            prompt = build_prompt(pred_class, confidence, patient_id, eye_side)
            report = generate_report(generator, prompt)
            # Save report
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"Image     : {img_name}\n")
                f.write(f"True class: {true_class}\n")
                f.write(f"Predicted : {pred_class} ({confidence:.1f}%)\n\n")
                f.write(report)
        else:
            # Load existing report
            if not os.path.exists(report_path):
                print(f"  Skipping {img_name} — report not found")
                continue
            with open(report_path, "r", encoding="utf-8") as f:
                lines  = f.readlines()
                report = "".join(lines[3:])   # skip header lines

        # Evaluate
        metrics = evaluate_report(report, pred_class, true_class)
        metrics['image']      = img_name
        metrics['pred_class'] = pred_class
        metrics['true_class'] = true_class
        metrics['confidence'] = round(confidence, 1)
        metrics['patient_id'] = patient_id
        all_results.append(metrics)
        class_scores[pred_class].append(metrics)

        # Print row
        safe_icon = "OK" if metrics['is_safe'] else "WARN"
        print(f"{i:<5} {img_name:<30} {true_class:<8} {pred_class:<8} "
              f"{metrics['bleu']:<6.3f} {metrics['rougeL']:<8.3f} "
              f"{metrics['bert_f1']:<6.3f} {metrics['clinical_acc']:<8.1f} {safe_icon}")

    # ==============================================================
    #  SUMMARY — mean ± std per class and overall
    # ==============================================================

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY — mean ± std per class")
    print(f"{'='*70}")
    print(f"{'Class':<10} {'N':<5} {'BLEU':<12} {'ROUGE-L':<12} {'BERTScore':<12} {'ClinAcc%':<12} {'Safe%'}")
    print(f"{'-'*70}")

    summary = {}
    for cls, results in class_scores.items():
        if not results:
            continue
        n        = len(results)
        bleus    = [r['bleu']        for r in results]
        rougeLs  = [r['rougeL']      for r in results]
        berts    = [r['bert_f1']     for r in results]
        clins    = [r['clinical_acc'] for r in results]
        safes    = [r['is_safe']     for r in results]

        summary[cls] = {
            'n':          n,
            'bleu':       (np.mean(bleus),   np.std(bleus)),
            'rougeL':     (np.mean(rougeLs), np.std(rougeLs)),
            'bert_f1':    (np.mean(berts),   np.std(berts)),
            'clinical_acc': (np.mean(clins), np.std(clins)),
            'safe_pct':   sum(safes) / n * 100,
        }

        print(f"{cls:<10} {n:<5} "
              f"{np.mean(bleus):.3f}±{np.std(bleus):.3f}  "
              f"{np.mean(rougeLs):.3f}±{np.std(rougeLs):.3f}  "
              f"{np.mean(berts):.3f}±{np.std(berts):.3f}  "
              f"{np.mean(clins):.1f}±{np.std(clins):.1f}    "
              f"{sum(safes)/n*100:.1f}%")

    # Overall
    all_bleus  = [r['bleu']         for r in all_results]
    all_rouges = [r['rougeL']       for r in all_results]
    all_berts  = [r['bert_f1']      for r in all_results]
    all_clins  = [r['clinical_acc'] for r in all_results]
    all_safes  = [r['is_safe']      for r in all_results]

    print(f"{'-'*70}")
    print(f"{'OVERALL':<10} {len(all_results):<5} "
          f"{np.mean(all_bleus):.3f}±{np.std(all_bleus):.3f}  "
          f"{np.mean(all_rouges):.3f}±{np.std(all_rouges):.3f}  "
          f"{np.mean(all_berts):.3f}±{np.std(all_berts):.3f}  "
          f"{np.mean(all_clins):.1f}±{np.std(all_clins):.1f}    "
          f"{sum(all_safes)/len(all_safes)*100:.1f}%")
    print(f"{'='*70}")

    # Hallucination report
    dangerous = [r for r in all_results if not r['is_safe']]
    print(f"\nHallucination check: {len(dangerous)} reports flagged out of {len(all_results)}")
    for r in dangerous:
        print(f"  Index: {r['image']} | Pred: {r['pred_class']} | "
              f"Dangerous keywords: {r['danger_keywords']}")

    # Save all results to JSON
    json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {json_path}")

    # Save summary to text file for paper
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Report Generation Evaluation Summary\n")
        f.write(f"Date: {date.today()}\n")
        f.write(f"Model: ViT-B IN-21k | LLM: Mistral-7B-Instruct-v0.3\n")
        f.write(f"Total reports evaluated: {len(all_results)}\n\n")
        f.write(f"{'Class':<10} {'N':<5} {'BLEU':<14} {'ROUGE-L':<14} {'BERTScore':<14} {'ClinAcc%':<14} {'Safe%'}\n")
        f.write("-" * 75 + "\n")
        for cls, s in summary.items():
            f.write(f"{cls:<10} {s['n']:<5} "
                    f"{s['bleu'][0]:.3f}±{s['bleu'][1]:.3f}    "
                    f"{s['rougeL'][0]:.3f}±{s['rougeL'][1]:.3f}    "
                    f"{s['bert_f1'][0]:.3f}±{s['bert_f1'][1]:.3f}    "
                    f"{s['clinical_acc'][0]:.1f}±{s['clinical_acc'][1]:.1f}      "
                    f"{s['safe_pct']:.1f}%\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'OVERALL':<10} {len(all_results):<5} "
                f"{np.mean(all_bleus):.3f}±{np.std(all_bleus):.3f}    "
                f"{np.mean(all_rouges):.3f}±{np.std(all_rouges):.3f}    "
                f"{np.mean(all_berts):.3f}±{np.std(all_berts):.3f}    "
                f"{np.mean(all_clins):.1f}±{np.std(all_clins):.1f}      "
                f"{sum(all_safes)/len(all_safes)*100:.1f}%\n")
    print(f"Summary saved to: {summary_path}")

    return summary


# ================================================================
#  EXPERT RATING HELPER
#  Run this separately after collecting ophthalmologist ratings.
#  Fill in ratings dict with scores 1-5 per image index.
# ================================================================

def compute_expert_ratings(ratings_dict):
    """
    ratings_dict format:
    {
        image_index: {
            'image': 'CH/image_1776.jpg',
            'pred_class': 'CH',
            'rating': 4        # ophthalmologist score 1-5
        },
        ...
    }
    """
    from collections import defaultdict
    class_ratings = defaultdict(list)

    for idx, data in ratings_dict.items():
        cls    = data['pred_class']
        rating = data['rating']
        class_ratings[cls].append(rating)

    all_ratings = [d['rating'] for d in ratings_dict.values()]

    print(f"\n{'='*45}")
    print("EXPERT RATING SUMMARY (1-5 scale)")
    print(f"{'='*45}")
    print(f"{'Class':<10} {'N':<5} {'Mean':<8} {'Std':<8} {'Min':<6} {'Max'}")
    print("-" * 45)

    for cls, ratings in sorted(class_ratings.items()):
        print(f"{cls:<10} {len(ratings):<5} "
              f"{np.mean(ratings):.2f}    "
              f"{np.std(ratings):.2f}    "
              f"{min(ratings):<6} {max(ratings)}")

    print("-" * 45)
    print(f"{'OVERALL':<10} {len(all_ratings):<5} "
          f"{np.mean(all_ratings):.2f}    "
          f"{np.std(all_ratings):.2f}    "
          f"{min(all_ratings):<6} {max(all_ratings)}")
    print(f"{'='*45}")


# ================================================================
#  RUN
# ================================================================

if __name__ == "__main__":
    print("Installing required packages if missing ...")
    os.system("pip install rouge-score bert-score nltk -q")

    import nltk
    nltk.download('punkt', quiet=True)

    summary = run_evaluation()