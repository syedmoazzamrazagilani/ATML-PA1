# ATML-PA1: Beyond IID Learning

**EE-5102 / CS-6304 — Advanced Topics in Machine Learning — Fall 2026**

This repository contains the code for Programming Assignment 1, which examines four aspects of learning beyond the standard IID closed-set setting: inductive biases and feature representations, unsupervised domain adaptation, domain generalization, and open-set recognition.

> **GitHub repository link is included at the end of the abstract in the submitted PDF report.**

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Repository Structure](#repository-structure)
- [Data Preparation](#data-preparation)
- [Task 1 — Inductive Biases and Feature Representations](#task-1--inductive-biases-and-feature-representations)
- [Task 2 — Unsupervised Domain Adaptation](#task-2--unsupervised-domain-adaptation)
- [Task 3 — Domain Generalization](#task-3--domain-generalization)
- [Task 4 — Open-Set Recognition](#task-4--open-set-recognition)
- [Reproducing Results](#reproducing-results)
- [External Code Attribution](#external-code-attribution)

---

## Environment Setup

Python 3.10+ and CUDA 11.8+ are recommended.

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Version |
|---------|---------|
| torch | ≥ 2.0.0 |
| torchvision | ≥ 0.15.0 |
| open-clip-torch | ≥ 2.20.0 |
| scikit-learn | ≥ 1.2.0 |
| umap-learn | ≥ 0.5.3 |
| timm | ≥ 0.9.0 |
| PyYAML | ≥ 6.0 |

---

## Repository Structure

```
ATML-PA1/
├── README.md
├── requirements.txt
├── .gitignore
│
├── shared/                         # Shared PACS data utilities (Tasks 2 & 3)
│   ├── pacs.py
│   ├── pacs_protocol.py
│   └── splits/
│       └── pacs_sketch_seed6304.json
│
├── task1/                          # Inductive Biases & Feature Representations
│   ├── configs/config.py
│   ├── data/
│   │   ├── make_subset.py
│   │   ├── make_cue_conflicts.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── backbones.py
│   │   └── train_heads.py
│   ├── analysis/
│   │   ├── evaluate_bias.py
│   │   ├── evaluate_color_bias.py
│   │   ├── evaluate_cue_conflicts.py
│   │   ├── evaluate_patch_shuffle.py
│   │   ├── evaluate_translation.py
│   │   ├── feature_similarity.py
│   │   ├── plot_translation.py
│   │   ├── representation.py
│   │   └── run_representation_analysis.py
│   ├── scripts/
│   │   └── run_task1.py
│   └── results/                    # populated at runtime
│
├── task2/                          # Unsupervised Domain Adaptation
│   ├── configs/
│   │   ├── base.yaml
│   │   ├── source_only.yaml
│   │   ├── dan.yaml
│   │   ├── dann.yaml
│   │   └── cdan.yaml
│   ├── models/
│   │   ├── backbone.py
│   │   └── domain_discriminator.py
│   ├── methods/
│   │   ├── source_only.py
│   │   ├── dan.py
│   │   ├── dann.py
│   │   └── cdan.py
│   ├── train.py
│   ├── evaluate_final.py
│   ├── controlled_study.py
│   └── results/                    # populated at runtime
│
├── task3/                          # Domain Generalization
│   ├── configs/
│   │   ├── base.yaml
│   │   ├── erm.yaml
│   │   ├── dan_dg.yaml
│   │   └── sam.yaml
│   ├── models/
│   │   └── backbone.py
│   ├── methods/
│   │   ├── erm.py
│   │   ├── dan_dg.py
│   │   └── sam.py
│   ├── selection/
│   │   └── source_validation.py
│   ├── evaluation/
│   │   ├── domain_metrics.py
│   │   ├── source_domain_separability.py
│   │   └── sharpness.py
│   ├── train.py
│   ├── evaluate_sketch.py
│   ├── controlled_study.py
│   └── results/                    # populated at runtime
│
└── task4/                          # Open-Set Recognition
    ├── configs/
    │   ├── vanilla.yaml
    │   ├── gcsc.yaml
    │   └── proser.yaml
    ├── data/
    │   ├── cifar10.py
    │   ├── cifar100_unknowns.py
    │   └── make_splits.py
    ├── models/
    │   └── resnet_cifar.py
    ├── methods/
    │   ├── vanilla.py
    │   ├── proser.py
    │   └── manifold_mixup.py
    ├── scores/
    │   ├── msp.py
    │   ├── mls.py
    │   ├── energy.py
    │   ├── mahalanobis.py
    │   └── proser_score.py
    ├── evaluation/
    │   ├── metrics.py
    │   ├── thresholds.py
    │   └── failure_analysis.py
    ├── cache/                       # populated by extract_outputs.py
    ├── extract_outputs.py
    ├── evaluate_osr.py
    ├── run_task4.py
    └── results/                     # populated at runtime
```

---

## Data Preparation

Raw datasets are **not committed** to this repository. Download and place them as follows.

### STL-10 (Task 1)
Downloaded automatically on first run:
```bash
python -m task1.data.make_subset
```
Saved to `data_cache/`.

### PACS (Tasks 2 & 3)
Download from the [PACS dataset page](https://domaingeneralization.github.io/#data) and extract so the structure is:
```
data_cache/PACS/
├── photo/
├── art_painting/
├── cartoon/
└── sketch/
```
Then generate the shared splits:
```bash
python -m shared.pacs_protocol
```
This creates `shared/splits/pacs_sketch_seed6304.json` with stratified 80/20 source splits using seed 6304.

### CIFAR-10 / CIFAR-100 (Task 4)
Downloaded automatically on first run:
```bash
python -m task4.data.make_splits
```
Saved to `data_cache/`.

---

## Task 1 — Inductive Biases and Feature Representations

**Dataset:** STL-10 (10 classes, 500-image evaluation subset, seed 6304)

**Models:** ResNet-50 (`IMAGENET1K_V2`), ViT-B/16 (`IMAGENET1K_V1`), CLIP ViT-B-32 (`openai`)

**Experimental design choices:**
- Color intervention: hue rotation (factor = 0.5) — changes chromatic identity while preserving luminance and spatial structure
- Visual rejection rule for cue conflicts: accept if `mean ∈ [0.05, 0.95]` AND `std > 0.05`, evaluated before any model sees the image
- Visualization: t-SNE with PCA pre-reduction to 50 dimensions, `perplexity=30`, `random_state=6304`

### Run all steps

```bash
# Full pipeline from scratch
python -m task1.scripts.run_task1

# Skip data download (splits already exist)
python -m task1.scripts.run_task1 --skip-data

# Skip cue-conflict generation (images already generated)
python -m task1.scripts.run_task1 --skip-data --skip-cue

# Skip representation analysis (no GPU / time constraint)
python -m task1.scripts.run_task1 --skip-data --skip-cue --skip-repr
```

### Run individual steps

```bash
# Step 0 — create data splits
python -m task1.data.make_subset

# Step 1 — train linear heads + clean baseline metrics
python -m task1.models.train_heads

# Step 2 — generate cue-conflict images (~45–90 min)
python -m task1.data.make_cue_conflicts

# Step 3 — color bias (grayscale + hue rotation)
python -m task1.analysis.evaluate_color_bias

# Step 4 — shape vs. texture (cue-conflict evaluation)
python -m task1.analysis.evaluate_cue_conflicts

# Step 5 — translation invariance
python -m task1.analysis.evaluate_translation

# Step 6 — patch structure (4×4 shuffle)
python -m task1.analysis.evaluate_patch_shuffle

# Step 7 — representation analysis (cosine stability + t-SNE)
python -m task1.analysis.run_representation_analysis

# Print combined summary of all results
python -m task1.analysis.evaluate_bias
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Seed | 6304 |
| Train/val split | 80/20 stratified |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Max epochs | 50 |
| Early-stop patience | 5 |
| Batch size | 64 |
| Test subset size | 500 (50 per class) |

### Outputs

All results written to `task1/results/`:

| File | Contents |
|------|----------|
| `clean_baseline.json` | Top-1 accuracy, macro-F1, mean confidence for all models |
| `color_bias.json` | Accuracy and prediction consistency for grayscale and hue rotation |
| `cue_conflicts.json` | Shape bias (%), coverage (%), per-pair breakdown, sample cases |
| `translation.json` | Per-shift accuracy and Consistency(δ) averaged over 4 directions |
| `patch_shuffle.json` | Accuracy drop and prediction consistency after 4×4 shuffle |
| `representation_stability.json` | Cosine stability IT for all interventions × all backbones |
| `translation_curve.png` | Accuracy and consistency vs. displacement (2-panel) |
| `tsne_<model>_<intervention>.png` | Joint t-SNE of clean + transformed features |

---

## Task 2 — Unsupervised Domain Adaptation

**Dataset:** PACS — Photo, Art Painting, Cartoon → Sketch (target, unlabeled during training)

**Model:** ResNet-18 (`IMAGENET1K_V1`), 7-class head, full fine-tuning

**BatchNorm policy:** Running means and variances frozen at ImageNet values for all methods. Scale (γ) and bias (β) remain trainable.

**Methods:** Source-only ERM, DAN (MMD, λ=1.0), DANN (adversarial), CDAN (class-conditional adversarial)

### Run

```bash
# Train all four methods (run in order — source_only checkpoint is required by Task 3)
python -m task2.train --method source_only
python -m task2.train --method dan
python -m task2.train --method dann
python -m task2.train --method cdan

# Final evaluation (target labels used here only)
python -m task2.evaluate_final

# Controlled alignment-strength study (DAN: λ ∈ {0.1, 1, 10})
python -m task2.controlled_study
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Seed | 6304 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Max epochs | 30 |
| Early-stop patience | 5 (mean source macro-F1) |
| Batch per source domain | 8 |
| Target batch size | 24 |
| MMD λ | 1.0 |
| MMD kernels | RBF × 3 (bandwidths: 0.5×, 1×, 2× median pairwise distance) |
| DANN discriminator | Linear(512→256) → ReLU → Dropout(0.5) → Linear(256→2) |
| Domain loss weight | 1.0 |

### Outputs

All results written to `task2/results/` and checkpoints to `task2/checkpoints/`.

| File | Contents |
|------|----------|
| `<method>_history.json` | Per-epoch classification loss, alignment loss, per-domain F1 |
| `final_results.json` | Source-val and target accuracy/F1 for all methods |
| `domain_separability.json` | Logistic-regression domain separability score per method |
| `class_analysis.json` | Per-class target accuracy and dominant confusions |
| `controlled_study_results.json` | DAN λ sweep results |

---

## Task 3 — Domain Generalization

**Dataset:** PACS — same source domains as Task 2, Sketch is fully unseen during training and model selection

**Important:** The ERM baseline here is the `source_only` checkpoint from Task 2 — it is copied, not retrained.

**No Sketch image or label is accessed until `evaluate_sketch.py`.**

**Methods:** ERM (loaded from Task 2), DAN-DG (pairwise source MMD, λ=1.0), SAM (ρ=0.05)

### Run

```bash
# Step 1 — ERM baseline (loads Task 2 source_only checkpoint)
python -m task3.train --method erm

# Step 2 — DAN-DG
python -m task3.train --method dan_dg

# Step 3 — SAM
python -m task3.train --method sam

# Final evaluation on Sketch (run only after all training is done)
python -m task3.evaluate_sketch

# Controlled study (λ_DG ∈ {0.1, 1, 10} or ρ ∈ {0.01, 0.05, 0.1})
python -m task3.controlled_study
```

### Hyperparameters

Identical to Task 2 except:

| Parameter | Value |
|-----------|-------|
| DAN-DG λ | 1.0 |
| MMD pairs | All 3 unordered source-domain pairs |
| SAM ρ | 0.05 |
| SAM base optimizer | AdamW (same lr/wd as ERM) |
| Sharpness proxy radius | 0.05 (normalized gradient ascent) |
| Sharpness batch | 32 per source domain, seed 6304 |

### Outputs

All results written to `task3/results/`:

| File | Contents |
|------|----------|
| `<method>_history.json` | Training curves (cls loss, MMD loss where applicable, source F1) |
| `sketch_results.json` | Per-method Sketch accuracy and macro-F1 |
| `source_separability.json` | 3-class logistic-regression separability for Photo/Art/Cartoon |
| `sharpness.json` | Sharpness proxy Δsharp for all three methods |
| `controlled_study_results.json` | λ or ρ sweep results |

---

## Task 4 — Open-Set Recognition

**Known classes:** CIFAR-10 (10 classes, 90/10 train/val split, seed 6304)

**Unknown classes (evaluation only, never seen during training):**
- Near: `bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel` (800 images)
- Far: `bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe` (800 images)

**Model:** CIFAR-adapted ResNet-18 (3×3 stride-1 first conv, no max-pool)

**Methods:** Vanilla (cross-entropy), GCSC (+ RandAugment), PROSER (classifier + data placeholders)

**Scores:** MSP, MLS, Energy, Mahalanobis (post-hoc on Vanilla); MLS + PROSER placeholder score for model comparison

### Run

```bash
# Step 1 — train Vanilla
python -m task4.methods.vanilla --method vanilla

# Step 2 — train GCSC
python -m task4.methods.vanilla --method gcsc

# Step 3 — cache features and logits (run for each method before evaluation)
python -m task4.extract_outputs --method vanilla
python -m task4.extract_outputs --method gcsc

# Step 4 — train PROSER (initialises from vanilla checkpoint)
python -m task4.methods.proser

# Step 5 — cache PROSER outputs
python -m task4.extract_outputs --method proser

# Step 6 — full OSR evaluation (tables, figures, failure analysis)
python -m task4.evaluate_osr
```

Or run everything in one go:
```bash
python -m task4.run_task4

# With flags if checkpoints/caches already exist
python -m task4.run_task4 --skip-train
python -m task4.run_task4 --skip-train --skip-extract
```

### Hyperparameters

**Vanilla / GCSC:**

| Parameter | Value |
|-----------|-------|
| Seed | 6304 |
| Optimizer | SGD (Nesterov) |
| Learning rate | 0.1 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| LR schedule | Cosine decay |
| Batch size | 128 |
| Epochs | 100 |
| Val split | 90/10 stratified |
| GCSC RandAugment | num_ops=2, magnitude=9 |

**PROSER fine-tuning:**

| Parameter | Value |
|-----------|-------|
| Init from | Vanilla best checkpoint |
| Learning rate | 1e-3 |
| Epochs | 50 |
| Dummy classifiers | 5 |
| Classifier-placeholder β | 1.0 |
| Data-placeholder γ | 0.1 |
| Mixup distribution | Beta(2, 2) |
| Mixup location | After layer2, before layer3 |

**Threshold calibration:** 95th percentile of unknownness on CIFAR-10 validation set (known data only).

### Outputs

All results written to `task4/results/`:

| File | Contents |
|------|----------|
| `<method>_history.json` | Per-epoch training loss and validation accuracy |
| `osr_results.json` | All AUROC, rejection rates, and failure cases |
| `osr_score_figure.png` | Score distributions + ROC curves (MSP, MLS, Mahalanobis) |
| `training_curves.png` | Training loss and val accuracy for all methods |

---

## Reproducing Results

To reproduce all results from scratch on a GPU machine (e.g. Kaggle):

```bash
# Add repo to path (Kaggle / notebook environments)
import sys; sys.path.insert(0, '/kaggle/working/ATML-PA1')

# Task 1
python -m task1.scripts.run_task1

# Task 2 (must complete before Task 3)
python -m task2.train --method source_only
python -m task2.train --method dan
python -m task2.train --method dann
python -m task2.train --method cdan
python -m task2.evaluate_final
python -m task2.controlled_study

# Task 3 (requires Task 2 source_only checkpoint)
python -m task3.train --method erm
python -m task3.train --method dan_dg
python -m task3.train --method sam
python -m task3.evaluate_sketch
python -m task3.controlled_study

# Task 4
python -m task4.run_task4
```

Every reported number traces to a JSON file in the corresponding `results/` directory. Every random operation uses seed 6304. Checkpoints are saved to `task*/checkpoints/` and are excluded from version control via `.gitignore`.

---

## External Code Attribution

| Component | Source |
|-----------|--------|
| Style-transfer loss (Task 1) | Adapted from [PyTorch Neural Style Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) |
| OpenCLIP model loading | [`mlfoundations/open_clip`](https://github.com/mlfoundations/open_clip) (Apache 2.0) |
| MMD kernel loss | Adapted from [DAN reference implementation](https://github.com/jindongwang/transferlearning) |
| Gradient-reversal layer | Adapted from [DANN reference implementation](https://github.com/fungtion/DANN) |
| PROSER loss | Adapted from [Zhou et al. (2021) reference code](https://github.com/zhoudw-zdw/CVPR21-Proser) |
