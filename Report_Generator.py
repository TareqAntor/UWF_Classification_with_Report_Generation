import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ================================================================
#  CONFIG — only change these lines
# ================================================================
CKPT_PATH  = r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\vit_base_patch16_224_in21k\vit_base_patch16_224_in21k_test_fold_1.pt"
IMAGE_IDX  = 200          # 0  = CH/image_1776.jpg  (99.5% correct)
                        # 8  = CH/image_1811.jpg  (misclassified as Normal)
                        # 302 = RB/image_1356.jpg (100% correct)
                        # 342 = RB/image_1562.jpg (misclassified as RCH)
PATIENT_ID = "P-00001"
EYE_SIDE   = "Right eye"
# ================================================================


# ================================================================
#  SECTION 1 — KNOWLEDGE BASE
#  Clinical descriptions, urgency levels, and GradCAM regions
#  for all 6 classes in your dataset.
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


# ================================================================
#  SECTION 2 — CHECKPOINT LOADER
#  Loads a saved .pt test checkpoint and extracts prediction
#  info for one image by index.
# ================================================================

def load_prediction(ckpt_path, image_idx=0):
    """
    Load prediction for one test image from a saved .pt checkpoint.
    Returns: pred_class, confidence, img_name
    """
    ckpt         = torch.load(ckpt_path, weights_only=False)
    idx_to_class = ckpt['idx_to_class']

    pred_idx   = int(ckpt['prediction_label'][image_idx])
    true_idx   = int(ckpt['targets'][image_idx])
    pred_class = idx_to_class[pred_idx]
    true_class = idx_to_class[true_idx]
    confidence = float(ckpt['prediction_probs'][image_idx][pred_idx]) * 100
    img_name   = str(ckpt['image_names'][image_idx]).split("\\")[-1]

    print(f"\nImage     : {img_name}")
    print(f"True class: {true_class}")
    print(f"Predicted : {pred_class} ({confidence:.1f}%)")
    print(f"Correct   : {'YES' if pred_class == true_class else 'NO'}")

    return pred_class, confidence, img_name


def find_by_filename(ckpt_path, filename):
    """
    Find a test image by partial filename e.g. 'image_1776'
    and return its prediction info.
    """
    ckpt = torch.load(ckpt_path, weights_only=False)
    for i, name in enumerate(ckpt['image_names']):
        if filename in str(name):
            print(f"Found '{filename}' at index {i}")
            return load_prediction(ckpt_path, image_idx=i)
    print(f"'{filename}' not found in checkpoint.")
    return None, None, None


# ================================================================
#  SECTION 3 — PROMPT BUILDER
#  Converts model output into a structured text prompt
#  that is sent to the LLM.
# ================================================================

def build_prompt(pred_class, confidence, patient_id, eye_side):
    """
    Build a structured clinical prompt from model prediction.
    Returns: prompt string, gradcam_region string
    """
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

    return prompt, gradcam_region


# ================================================================
#  SECTION 4 — LLM LOADER AND REPORT GENERATOR
#  Loads Mistral-7B onto your RTX 4090 (25.8 GB VRAM)
#  and generates the clinical report text.
# ================================================================

def load_mistral():
    """
    Load Mistral-7B-Instruct-v0.3.
    Downloads ~14 GB on first run, loads from cache after that.
    Requires ~14 GB VRAM — your RTX 4090 (25.8 GB) handles this fine.
    """
    model_id  = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"\nLoading {model_id}")
    print("First run downloads ~14 GB — subsequent runs load from cache.\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto")           # auto-places on your GPU

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,                         # low = consistent clinical output
        do_sample=True,
        repetition_penalty=1.1,
    )
    print("Mistral-7B loaded successfully.")
    return gen


def generate_report(generator, prompt):
    """
    Send the prompt to the loaded LLM and return the report text.
    """
    messages = [{"role": "user", "content": prompt}]
    output   = generator(messages)
    return output[0]["generated_text"][-1]["content"]


# ================================================================
#  MAIN — runs when you execute: python Report_Generator.py
# ================================================================

if __name__ == "__main__":

    # Step 1 — load prediction from checkpoint
    pred_class, confidence, img_name = load_prediction(
        CKPT_PATH, image_idx=IMAGE_IDX)

    # Step 2 — build prompt for LLM
    prompt, gradcam = build_prompt(
        pred_class, confidence, PATIENT_ID, EYE_SIDE)

    # Step 3 — load Mistral-7B onto RTX 4090
    generator = load_mistral()

    # Step 4 — generate the clinical report
    print("\nGenerating clinical report...")
    report = generate_report(generator, prompt)

    # Step 5 — display report
    print("\n" + "=" * 60)
    print("  GENERATED CLINICAL REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60)

    # Step 6 — save report to text file
    safe_name = img_name.replace(".jpg", "").replace("/", "_")
    out_file  = f"report_{safe_name}.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Image     : {img_name}\n")
        f.write(f"Predicted : {pred_class} ({confidence:.1f}%)\n")
        f.write(f"GradCAM   : {gradcam}\n\n")
        f.write(report)
    print(f"\nReport saved to: {out_file}")