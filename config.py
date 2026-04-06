# ==============================================================================
#  config.py  —  Ultra-Wide-Field Fundus  |  Multi-Class (6 classes)
#
#  PAPER BASELINE (ViT-B to beat):
#    Accuracy: 91.46  |  AUC: 96.87  |  Precision: 82.87
#    Sensitivity: 81.37  |  F1: 81.42  |  Specificity: 97.60  |  Kappa: 84.14
#
#  RECOMMENDED EXPERIMENT ORDER  (all available in your models.py):
#
#  Priority  model_to_load                     Why
#  ────────  ────────────────────────────────  ──────────────────────────────────
#  🥇 1st   vit_base_patch16_224_in21k         Same arch as paper's ViT-B but
#                                              pretrained on 21k classes instead
#                                              of 1k → stronger features on
#                                              small medical datasets
#
#  🥈 2nd   vit_base_patch8_224_in21k          Finer patches (8px vs 16px) =
#                                              better capture of fine lesion
#                                              textures in fundus images
#
#  🥉 3rd   convnext_base_in22k                Paper only tried ConvNeXt-Tiny.
#                                              Base variant + 22k pretraining
#                                              is significantly stronger
#
#  4th      convnext_large_in22k               Even larger ConvNeXt, worth
#                                              trying if GPU memory allows
#
#  5th      vit_base_patch16_384               Higher resolution (384×384)
#                                              captures peripheral retinal
#                                              details better — UWF images
#                                              have important peripheral info
#
#  6th      vit_base_r50_s16_224_in21k         Hybrid ResNet50 + ViT backbone.
#                                              ResNet local features + global
#                                              attention. Great for small data.
#
#  7th      convnext_base_384_in22ft1k         ConvNeXt-Base fine-tuned at
#                                              384 resolution on IN-1k after
#                                              IN-22k pretraining. Best of
#                                              both worlds.
#
#  8th      mvitv2_base                        Multiscale ViT — hierarchical
#                                              attention at multiple scales,
#                                              well-suited for detecting tumors
#                                              of varying sizes
#
#  9th      efficientnet_b4                    Classic strong performer for
#                                              medical imaging, input 380×380
#
#  10th     vit_base_patch16_224_miil_in21k    MIIL pretraining on IN-21k with
#                                              different label space — sometimes
#                                              transfers better to medical data
#
#  ── RESIZE GUIDE ────────────────────────────────────────────────────────────
#  224×224  →  all vit_*_224, convnext_base/large, mvitv2_*, efficientnet_b0-b3
#  384×384  →  vit_base_patch16_384, convnext_*_384_*, convnext_*_in22ft1k
#  380×380  →  efficientnet_b4
#  456×456  →  efficientnet_b5
#
# ==============================================================================

config = {

    # ── Paths ─────────────────────────────────────────────────────────────────
    'parentdir':        r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Data_384/",
    'dataset_location': r"D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Data_384/",
    'Results_path':     r'D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results',
    'save_path':        r'D:\Tareq\Ultra-Wide-Field-Fundus-Image-Dataset\Results\convnext_base_384_in22ft1k',

    # ── Model selection ───────────────────────────────────────────────────────
    # Change 'model_to_load' and 'model_name' together for each experiment.
    # 'model_name' is only used for naming saved result files — make it unique.
    'model_to_load': 'convnext_base_384_in22ft1k',   # ← change per experiment
    'model_name':    'convnext_base_384_in22ft1k',   # ← change per experiment

    # ── Pretraining ───────────────────────────────────────────────────────────
    # Always True — use pretrained ImageNet weights, never train from scratch
    # on a 2000-image dataset.
    'ImageNet': True,

    # ── Input ─────────────────────────────────────────────────────────────────
    'input_ch':   3,             # RGB fundus images
    'Resize_h':   384,           # ← set to 384 for vit_*_384 / convnext_*_384
    'Resize_w':   384,           # ← set to 384 for vit_*_384 / convnext_*_384

    # ImageNet normalisation (correct for all pretrained models)
    'input_mean': [0.485, 0.456, 0.406],
    'input_std':  [0.229, 0.224, 0.225],

    # ── Training ──────────────────────────────────────────────────────────────
    'batch_size':      16,       # reduce to 8 for 384-input or large models
    'optim_fc':        'Adam',   # 'Adam' or 'SGD'
    'lr':              2e-5,     # low LR is key for fine-tuning transformers
    'n_epochs':        50,
    'stop_criteria':   'loss',   # 'loss' or 'accuracy'
    'max_epochs_stop': 10,       # early-stop patience (epochs)
    'epochs_patience': 5,        # ReduceLROnPlateau patience
    'lr_factor':       0.5,      # LR multiplier when plateau hit

    # ── Augmentation (used in TrainCNN transforms) ────────────────────────────
    'RotaionDegree': 20,
    'RHFlip':        0.5,
    'P_padding':     0,
    'P_fill':        0,
    'P_padding_mode':'constant',

    # ── Folds ─────────────────────────────────────────────────────────────────
    'num_folds':   5,
    # Set fold_to_run=[] to run all 5 folds.
    # Set fold_to_run=[1, 1] to run only fold 1 (useful for quick testing).
    'fold_to_run': [5,5],

    # ── ONN (Operational Neural Network) settings ─────────────────────────────
    # Keep ONN=False for standard pretrained models.
    # Set ONN=True only if using SelfONN_* models.
    'ONN':     False,
    'q_order': None,

    # ── Load existing model ───────────────────────────────────────────────────
    # Set to False to train from scratch (i.e. from ImageNet weights).
    # Set to a .pt file path to resume from a saved checkpoint.
    'load_model': False,
    'encoder':    False,
}


# ==============================================================================
#  QUICK SWAP TABLE — copy-paste the three lines below to switch experiments
#  Just change model_to_load, model_name, save_path, and resize if needed.
#
#  Experiment 1 — ViT-B IN-21k (START HERE)
#  'model_to_load': 'vit_base_patch16_224_in21k',
#  'model_name':    'vit_base_patch16_224_in21k',
#  'Resize_h': 224, 'Resize_w': 224,
#
#  Experiment 2 — ViT-B patch8 IN-21k
#  'model_to_load': 'vit_base_patch8_224_in21k',
#  'model_name':    'vit_base_patch8_224_in21k',
#  'Resize_h': 224, 'Resize_w': 224,
#
#  Experiment 3 — ConvNeXt-Base IN-22k
#  'model_to_load': 'convnext_base_in22k',
#  'model_name':    'convnext_base_in22k',
#  'Resize_h': 224, 'Resize_w': 224,
#
#  Experiment 4 — ConvNeXt-Large IN-22k
#  'model_to_load': 'convnext_large_in22k',
#  'model_name':    'convnext_large_in22k',
#  'Resize_h': 224, 'Resize_w': 224,
#
#  Experiment 5 — ViT-B 384 resolution
#  'model_to_load': 'vit_base_patch16_384',
#  'model_name':    'vit_base_patch16_384',
#  'Resize_h': 384, 'Resize_w': 384,   ← MUST change resize
#  'batch_size': 8,                     ← reduce batch for 384
#
#  Experiment 6 — Hybrid ViT (ResNet50 + ViT)
#  'model_to_load': 'vit_base_r50_s16_224_in21k',
#  'model_name':    'vit_base_r50_s16_224_in21k',
#  'Resize_h': 224, 'Resize_w': 224,
#
#  Experiment 7 — ConvNeXt-Base 384 (fine-tuned from 22k)
#  'model_to_load': 'convnext_base_384_in22ft1k',
#  'model_name':    'convnext_base_384_in22ft1k',
#  'Resize_h': 384, 'Resize_w': 384,   ← MUST change resize
#  'batch_size': 8,
#
#  Experiment 8 — MViTv2-Base
#  'model_to_load': 'mvitv2_base',
#  'model_name':    'mvitv2_base',
#  'Resize_h': 224, 'Resize_w': 224,
#
#  Experiment 9 — EfficientNet-B4
#  'model_to_load': 'efficientnet_b4',
#  'model_name':    'efficientnet_b4',
#  'Resize_h': 380, 'Resize_w': 380,   ← MUST change resize
#
#  Experiment 10 — ViT-B MIIL IN-21k
#  'model_to_load': 'vit_base_patch16_224_miil_in21k',
#  'model_name':    'vit_base_patch16_224_miil_in21k',
#  'Resize_h': 224, 'Resize_w': 224,
# ==============================================================================
