# DERMA-ACT V2 — Trustworthy Active Skin Lesion Assessment

**DERMA-ACT V2** is a research framework for seven-class dermoscopic skin lesion classification, finite-horizon sequential virtual-view evaluation, probability calibration, and split-conformal / Mondrian uncertainty quantification on the retrospective ISIC 2019 benchmark.

> **Research Software Only — Not for Clinical Use:** This repository accompanies a retrospective computational comparative study. It is **not** a cleared medical device, **not** clinically validated, **not** production-ready, and **must not** be used for clinical diagnosis, triage, or patient treatment.

---

## 1. Project Title and Overview

**Project:** DERMA-ACT V2 — Event-Aware Sequential Virtual-View Policy Evaluation and Uncertainty Quantification for Dermoscopic Image Classification  
**Author:** Duc-Son Le (ORCID: [0009-0000-5804-7811](https://orcid.org/0009-0000-5804-7811))  
**Public Code Repository:** [https://github.com/noenc207/DERMA-ACT](https://github.com/noenc207/DERMA-ACT)

DERMA-ACT V2 investigates whether an event-aware sequential view-acquisition policy operating over deterministic virtual views (`A1`–`A9`) of a single dermoscopic photograph improves seven-class diagnostic accuracy, uncertainty calibration, or decision-stability ranking relative to both a single global view (`A0`) and fixed-sequence comparators under explicit operational, statistical, and finite-sample boundaries.

---

## 2. Research Scope

- **Benchmark Cohort:** Retrospective **ISIC 2019 Training Set** (`N = 25,331` dermoscopic images across `13,931` operational grouping units; `7`-class formulation merging `AK` [`867`] + `SCC` [`628`] into `AKIEC` [`1,495`]).
- **Evaluation Design:** Group-disjoint Train (`16,702`), Validation (`3,542`), Calibration (`1,401`, split into `D_fit = 698` and `D_conf = 703` for split-conformal calibration), and Held-Out Test (`3,686` images; `K = 1,972` operational grouping units).
- **Out-of-Scope Boundaries:** No independent external clinical validation, no human-reader study, and no prospective bedside evaluation have been completed or claimed.

---

## 3. Current Project Status

- **Software Status:** Experimental research software / reproducible computational audit prototype.
- **Manuscript Status:** Pre-submission manuscript revision **`V3.2.1`** prepared for *Biomedical Signal Processing and Control* (Elsevier); not yet submitted or peer-reviewed. Unpublished manuscript drafts and submission documents are maintained privately outside the public repository.
- **Release Status:** Audited clean-room source code, configurations, pure aggregate benchmark tables/summaries, documentation, and curated Jupyter tutorials are published at [https://github.com/noenc207/DERMA-ACT](https://github.com/noenc207/DERMA-ACT). Heavy weights (`.ckpt`, `.pt`), per-image `.npz` caches, and restricted `CC-BY-NC 4.0` raw imagery are excluded from Git distribution.

---

## 4. Scientific Objectives

1. **Benchmark Single-View Discrimination and Calibration (`A0`):** Establish reproducible seven-class accuracy, balanced accuracy, Macro OvR AUROC, ECE, NLL, and Brier score for a frozen `EfficientNet-B4` classifier with post-hoc temperature scaling.
2. **Evaluate Sequential Virtual-View Policies (`A1`–`A9`):** Compare a finite-horizon supervised surrogate policy (`Learned Policy V1`) against non-adaptive fixed-sequence comparators (`Val-Ranked Fixed CAL-Stop`, `Val-Ranked Unstopped`, `Index-Fixed`), `Greedy Min-Entropy`, and a controlled post-hoc five-spatial-action ablation (`{A1, A3, A4, A5, A6}`).
3. **Audit Operator Fidelity and Prediction Stability:** Quantify deterministic transform flaws (`A7` Gabor saturation/collapse; `A9` non-semantic morphological masking), evaluate the pre-acquisition decision-flip head across realized on-policy transitions, and measure controlled digital perturbation sensitivity on a held-out validation subset.
4. **Quantify Finite-Sample Conformal Coverage:** Audit pooled split-conformal LAC and image-level class-conditional Mondrian LAC sets across image-level, cluster-level, and minority-class (`AKIEC`, `DF`, `VASC`) strata under honest grouping assumptions.

---

## 5. Methodology Overview

```text
Input Dermoscopic Image (A0: Resize 291 -> CenterCrop 224 -> ImageNet Norm)
   │
   ▼
Frozen EfficientNet-B4 Backbone (17.9M params, 7-class logits)
   ├──► Post-Hoc Temperature Scaling (T_full = 0.851796; T_fit = 0.855690)
   │       └──► Split-Conformal & Image-Level Mondrian LAC Sets (alpha = 0.10)
   └──► Initial State s_0 in R^28: [p_t(7), H(1), max_p(1), margin(1), history(9), mask(9)]
           │
           ▼  (up to budget B = 5 virtual views from A1–A9)
   MultiHeadActionPolicy (6,787 params; supervised surrogate MLP)
     ├── Predicts (Delta Q_hat, Delta C_hat, flip_logit) per candidate action
     ├── Computes utility U = 5.0*Delta Q_hat + 1.0*Delta C_hat - 0.5*sigma(logit) - 0.25
     ├── Stops if max_a U(s_t, a) < tau (-3/7 = -0.428571...)
     └── Otherwise applies deterministic view operator -> Resize(224) -> Backbone
           └── Recursive Arithmetic Probability Fusion: p_{t+1} = (b*p_t + p_cand)/(b+1)
```

- **State & Fusion (`research/derma_act/core/`):** Explicit 28-D state (`EPS = 1e-12`) with `history[k] + mask[k] = 1` invariant and recursive arithmetic running mean (`fuse_posterior`) across views of the **same** photograph — **not** Bayesian posterior inference.
- **Surrogate Policy (`src/modules/active_perception/policy.py`):** 2-layer MLP (`37 -> 64 -> 64` + 3 scalar linear heads, `6,787` parameters) trained offline on one-step supervision targets (`Delta Q`, `Delta C`, binary flip `I`) — a finite-horizon supervised surrogate value estimator, **not** a Bellman-optimized POMDP.

---

## 6. Seven-Class ISIC 2019 Classification

Raw ISIC 2019 ground truth contains eight non-zero diagnostic categories (`MEL`, `NV`, `BCC`, `AK`, `BKL`, `DF`, `VASC`, `SCC`) plus zero-count `UNK`. DERMA-ACT V2 maps `AK` (`867`) + `SCC` (`628`) into `AKIEC` (`1,495`), yielding seven canonical classes (`research/derma_act/config.py`):

| Index | Code | Class Name | Total (`N`) | Train | Val | Cal (`D_fit` + `D_conf`) | Held-Out Test |
| :---: | :---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 0 | `MEL` | Melanoma | 4,522 | 2,991 | 611 | 234 (133 + 101) | 686 |
| 1 | `NV` | Melanocytic Nevi | 12,875 | 8,461 | 1,819 | 724 (341 + 383) | 1,871 |
| 2 | `BCC` | Basal Cell Carcinoma | 3,323 | 2,211 | 444 | 202 (99 + 103) | 466 |
| 3 | `AKIEC` | Actinic Keratosis / Intraepithelial Carcinoma | 1,495 | 985 | 226 | 63 (33 + 30) | 221 |
| 4 | `BKL` | Benign Keratosis | 2,624 | 1,697 | 385 | 144 (77 + 67) | 398 |
| 5 | `DF` | Dermatofibroma | 239 | 175 | 24 | 15 (6 + 9) | 25 |
| 6 | `VASC` | Vascular Lesions | 253 | 182 | 33 | 19 (9 + 10) | 19 |
| — | **Total** | **Images (`N`) / Operational Groups (`K`)** | **25,331 / 13,931** | **16,702 / 9,198** | **3,542 / 1,972** | **1,401 / 789** | **3,686 / 1,972** |

- **Backbone Protocol (`supplementary/tables/tab_s1_classifier_protocol.tex`):** `timm` `efficientnet_b4.ra2_in1k` (`17.9M` params), `BatchNorm1d(1792) -> Dropout(0.3) -> Linear(1792, 7)`, `WeightedClassSampler`, `CrossEntropyLoss(label_smoothing=0.1)`, `AdamW(lr=1e-4, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=50)`, AMP `16-mixed`, selected at epoch `9` (`10` epochs run; `5,220` steps; validation AUROC `0.9331`). Note: static YAML defaults in `configs/train.yaml` (`lr=1e-3`, `loss=focal`) differ from the verified executed training code; see [configs/README.md](configs/README.md).

---

## 7. Sequential Spatial-View Analysis

Candidate operators `A1`–`A9` are deterministic software transformations applied to the **same** static input image (`src/modules/active_perception/view_generator.py`), **not** physical camera re-acquisitions:

| Token | Operator Definition | Category | Known Audit Status / Caveat |
| :---: | :--- | :--- | :--- |
| `A0` | Global baseline view (`Resize(291) -> CenterCrop(224)`) | Baseline | Unmodified reference view (`100%` of episodes) |
| `A1` | Center Zoom (`60%` central crop -> `224x224`) | Spatial | Valid; selected in `90.72%` of V1 test episodes |
| `A2` | Central-Mask Border (`40%` center zeroed -> `224x224`) | Spatial | Valid; lower standalone accuracy (`51.68%`); excluded from fixed comparator |
| `A3` | Upper Region (top `50%` -> `224x224`) | Spatial | Valid; included in fixed comparator |
| `A4` | Lower Region (bottom `50%` -> `224x224`) | Spatial | Valid; included in fixed comparator |
| `A5` | Left Region (left `50%` -> `224x224`) | Spatial | Valid; included in fixed comparator |
| `A6` | Right Region (right `50%` -> `224x224`) | Spatial | Valid; included in fixed comparator |
| `A7` | High-Frequency Gabor Texture (`cv2.filter2D` `CV_8UC3`) | Frequency | **Documented flaw:** 8-bit clipping causes `99.97%` median saturation & `25.69%` (`910/3,542`) zero-collapse on Val |
| `A8` | Shades-of-Gray Color Constancy ($p = 6$) | Color | Legacy display token `'Color Suppressed'` in `config.py`; selected in `74.96%` of V1 episodes |
| `A9` | Morphological Black-Hat (`17x17`, thresh `10`) + Telea Inpaint | Morphological | **Non-semantic dark-structure mask** (`11.55%` mean Val area); **not** a semantic hair detector |

- **Fixed Comparator Inventory:** Validation-ranked sequence `A6 -> A5 -> A3 -> A1 -> A4` accesses only five spatial actions (`{A1, A3, A4, A5, A6}`), whereas full `Learned Policy V1` accesses `A1`–`A9`.
- **Restricted Five-Spatial-Action Ablation (`Phase F`):** Post-hoc candidate masking restricts frozen `policy.pt` at inference to `{A1, A3, A4, A5, A6}` while keeping the training invariant `history[k] + mask[k] = 1` intact. This controls test-time action-inventory asymmetry without retraining `policy.pt` from scratch.

---

## 8. Calibration and Conformal Prediction

- **Post-Hoc Temperature Scaling:** BFGS NLL minimization on frozen `A0` calibration logits yields $T_{\mathrm{full}} = 0.851796$ on full $\mathcal{D}_{\mathrm{cal}}$ ($N = 1,401$) and $T_{\mathrm{fit}} = 0.855690$ on $\mathcal{D}_{\mathrm{fit}}$ ($N = 698$) for leak-free split-conformal scoring.
- **Pooled Split-Conformal LAC (`alpha = 0.10`):** Calibrated on $\mathcal{D}_{\mathrm{conf}}$ ($N = 703$, $K = 396$ operational groups) using exact order statistic $k^* = \lceil (703 + 1)(1 - 0.10) \rceil = 634$, giving threshold $q^* = 0.904267$ (probability inclusion cutoff $1 - q^* = 0.095733$).
- **Image-Level Mondrian LAC (`alpha = 0.10`):** Computes class-conditional order statistics $k_c = \lceil (n_c + 1)(1 - \alpha) \rceil$ per class on $\mathcal{D}_{\mathrm{conf}}$. Because calibration strata are small for `AKIEC` ($n_c = 30$, $K_c = 17$), `DF` ($n_c = 9$, $K_c = 4$), and `VASC` ($n_c = 10$, $K_c = 5$), `DF` and `VASC` hit the sample maximum ($k_c = n_c$) and cannot support cluster-level quantiles ($K_c < 9$), while `AKIEC` exhibits substantial finite-sample test under-coverage (`70.59%`).

---

## 9. Verified Experimental Results

All metrics below are verified directly against canonical aggregate artifacts under `research/derma_act/` and `supplementary/tables/` on held-out ISIC 2019 Test ($N = 3,686$ images; $K = 1,972$ operational groups; $B = 2,000$ cluster-stratified bootstrap 95% CIs, `seed = 20261112`).

### A. Single Global-View Baseline (`A0`, `tab2_baseline_performance.tex`)

| Metric | Raw `A0` Probabilities [95% Cluster CI] | Temperature-Scaled ($T_{\mathrm{full}} = 0.851796$) | Unit / Notes |
| :--- | :---: | :---: | :--- |
| Top-1 Accuracy | **69.21%** [66.95, 71.42] (`2,551 / 3,686`) | **69.21%** [66.95, 71.42] | Image-level ($N = 3,686$) |
| Top-1 Error (Risk) | **30.79%** [28.58, 33.05] | **30.79%** [28.58, 33.05] | Image-level ($N = 3,686$) |
| Balanced Accuracy | **68.83%** [64.35, 72.51] | **68.83%** [64.35, 72.51] | 7-class macro mean |
| Macro OvR AUROC | **0.9274** [0.9156, 0.9380] | **0.9274** [0.9159, 0.9380] | Image-level ($N = 3,686$) |
| 15-Bin ECE | **2.52%** [1.88, 4.75] | **2.71%** [1.90, 4.98] | Slight test over-sharpening ($T < 1$) |
| Negative Log-Likelihood | **0.8273** [0.7813, 0.8753] | **0.8174** [0.7656, 0.8727] | Nats / image (`-0.0099`) |
| Multi-Class Brier Score | **0.4207** [0.3958, 0.4466] | **0.4210** [0.3941, 0.4492] | Squared probability error |

### B. Sequential Strategy Comparison at Budget `B = 5` (Kept Separate by Action Inventory)

| Experiment Track | Strategy | Action Pool | Mean Views | Top-1 Risk [95% Cluster CI] | Top-1 Acc | Bal. Acc | Early Stop | Holm $p_{\mathrm{adj}}$ |
| :--- | :--- | :---: | ---: | :---: | ---: | ---: | ---: | :---: |
| **Baseline** | Global Baseline (`A0`, `B = 0`) | `{A0}` | 0.00 | 30.79% [28.58, 33.05] | 69.21% | 68.83% | 100.00% | $\le 1.64\times 10^{-4}$ vs V1 (`B=1..5`) |
| **Full 9-Action** | Learned Policy V1 (`A1`–`A9`) | 9 actions | 4.02 | 26.94% [24.72, 29.30] | 73.06% | 70.04% | 33.61% | Reference ($m = 25$ family) |
| **Fixed Comparator** | Val-Ranked Fixed (`CAL-Stop`) | 5 spatial | 3.90 | 25.96% [23.66, 28.35] | 74.04% | 67.63% | 22.65% | `0.3185` vs V1 (non-significant; **not** equivalence) |
| **Fixed Comparator** | Val-Ranked Fixed (Unstopped) | 5 spatial | 5.00 | 25.88% [23.58, 28.28] | 74.12% | 67.66% | 0.00% | `0.2070` vs V1 (non-significant; **not** equivalence) |
| **Fixed Comparator** | Index-Fixed (`A1`–`A5`) | 5 actions | 5.00 | 26.70% | 73.30% | 68.13% | 0.00% | `1.0000` vs V1 |
| **Heuristic** | Greedy Min-Entropy | 9 actions | 5.00 | 27.27% | 72.73% | 70.35% | 0.00% | `1.0000` vs V1 |
| **Restricted Ablation** | Spatial-Masked V1 (`Phase F`) | 5 spatial | 3.44 | 26.21% [23.99, 28.52] | 73.79% | 69.05% | 45.85% | `1.0000` vs CAL-Stop; `0.2387` vs 9-Act V1 ($m=20$) |

> **Critical Statistical Interpretation:** Non-significant Holm–Bonferroni adjusted $p$-values (`p_adj = 0.3185`, `0.2070`, `1.0000`) indicate absence of statistically detected difference at $\alpha = 0.05$ and **do not establish formal statistical equivalence** or policy optimality. At `B = 5`, full V1 vs `CAL-Stop` exhibits class-sensitivity trade-offs (`AKIEC` `+9.05 pp`, `p_adj = 0.0016` vs `NV` `-3.10 pp`, `p_adj = 6.80e-11` and `BKL` `-5.78 pp`, `p_adj = 0.0011`; $m = 7$ family).

### C. Decision-Flip Ranking & Conformal Prediction Summary

- **Decision-Flip Instability Head (`tab5_decision_flip_audit.tex`):** Evaluated across $M = 14,815$ realized on-policy test transitions ($988$ Top-1 flips = `6.67%` prevalence across $K = 1,887$ multi-view groups): **ROC AUROC `0.9473`** [0.9417, 0.9525], **Step-Function AP `0.5172`** [0.4819, 0.5509], **Brier `0.0416`** [0.0381, 0.0453]. No operational runtime alert threshold has been prespecified or validated.
- **Conformal Coverage Estimands (`tab6_conformal_summary.tex` & `tab10_mondrian_conformal.tex`, $\alpha = 0.10$):**
  - *Image-level marginal coverage (Pooled LAC, $N = 3,686$):* **`90.67%`** [89.14, 92.12], mean set size **`1.925`** [1.882, 1.970], singleton rate `38.09%`.
  - *Group-level estimands (Pooled LAC, $K = 1,972$):* Equal-cluster-weighted mean **`92.80%`** [91.71, 93.80]; strict all-images-covered conjunction **`89.76%`** (`1,770 / 1,972`); any-image-covered **`94.83%`** (`1,870 / 1,972`).
  - *Subgroup image coverage (Pooled LAC):* Singleton-group images ($k = 1$, $N = 1,226$): **`94.94%`**; multi-image-group photos ($k \ge 2$, $N = 2,460$): **`88.54%`**; `AKIEC` ($N = 221$): **`74.21%`** [cluster CI: 64.73, 83.16; pointwise Wilson CI: 68.06, 79.53].
  - *Image-level Mondrian LAC ($N = 3,686$):* Marginal image coverage **`90.64%`** [89.21, 92.04], equal-cluster mean **`93.12%`**, mean set size **`2.390`** [2.334, 2.448], singleton rate `27.56%`, worst-class coverage **`70.59%`** (`AKIEC`, pointwise image-level Wilson CI `[64.3, 76.2]`).
- **Digital Perturbation Sensitivity (`tab7_perturbation_robustness.tex`):** Evaluated on $N = 50$ held-out Validation images across `16` transformations on `A0` (`800` observations): spatial shifts/crops show the highest shift (5% center crop: `92.0%` Top-1 consistency, mean TVD `0.0862`; ±5 px translations: `90.0%–92.0%`, TVD `0.0784–0.0813`), whereas ±5% photometric/gamma adjustments retain `96.0%–100.0%` consistency (TVD `0.0127–0.0233`).

---

## 10. Research Limitations

All users and downstream readers must respect the following verified scientific and methodological constraints (detailed further in [docs/LIMITATIONS.md](docs/LIMITATIONS.md)):

1. **No Patient-Level Independence:** ISIC 2019 omits patient identifiers (`patient_id`). Operational grouping uses observed `lesion_id` where available (`23,247` images → `11,847` groups) and `image_id` singletons otherwise (`2,084` images), totaling `13,931` operational groups. Patient-level independence across distinct lesions has **not** been established.
2. **Supervised Surrogate, Not Bellman POMDP:** `MultiHeadActionPolicy` is a finite-horizon supervised surrogate estimator predicting one-step deltas (`Delta Q`, `Delta C`, `I`), not a Bellman-optimized POMDP solver.
3. **Arithmetic Mean Fusion, Not Bayesian Inference:** Sequential evidence aggregation computes an unweighted arithmetic running mean (`fuse_posterior`) over deterministically transformed crops of a single static photograph, not Bayesian posterior updating.
4. **`A7` Gabor Saturation / Collapse Flaw:** `A7` applies `cv2.filter2D` with `CV_8UC3` clipping prior to max pooling, producing `99.97%` median pixel saturation and `25.69%` (`910 / 3,542`) all-zero normalized collapse on Validation.
5. **`A9` Non-Semantic Masking:** `A9` applies deterministic `17x17` black-hat thresholding (`> 10`) and Telea inpainting (`11.55%` mean masked area on Validation); it is **not** a semantic hair detector.
6. **Decision-Flip Head Scope:** The instability head ranks the probability that the next realized on-policy transition changes Top-1 `argmax`; it is **not** a validated clinical alert or error-prevention trigger.
7. **Small-Sample `AKIEC` Conformal Under-Coverage:** Neither pooled LAC (`74.21%`) nor image-level Mondrian LAC (`70.59%`, calibrated on only `n_c = 30` images across `K_c = 17` clusters) achieves 90% empirical coverage on `AKIEC`.
8. **Clustering Caveat on Wilson CIs:** Image-level Wilson confidence intervals reported for per-class conformal strata do **not** automatically account for within-lesion clustering.
9. **No External, Reader, or Prospective Clinical Validation:** No independent external cohort validation, no human-reader validation, and no prospective clinical study have been completed.
10. **Research Software Only:** This codebase is strictly research software and is **not** intended for clinical diagnosis, screening, or treatment.

---

## 11. Repository Structure

```text
.
├── README.md                     # Project overview, verified metrics, setup & reproduction index
├── LICENSE_STATUS.md             # Layer-by-layer legal & CC-BY-NC 4.0 governance posture
├── CITATION.cff                  # Citation metadata
├── pyproject.toml                # Python package & pytest configuration
├── requirements.txt              # Core ML/DL dependencies
├── requirements-lock.txt         # Host freeze snapshot
├── requirements-notebooks.txt    # Pinned optional Jupyter notebook toolchain
├── src/                          # Core active perception, selective decision & training modules (src/README.md)
├── research/                     # DERMA-ACT canonical math, frozen configs & aggregate summaries (research/README.md)
├── scripts/                      # Portable evaluation, table & figure builders (scripts/README.md)
├── configs/                      # Hydra YAML configuration presets (configs/README.md)
├── tests/                        # Clean-room math parity & repository hygiene tests (tests/README.md)
├── notebooks/                    # Curated 7-notebook tutorial & audit suite (notebooks/README.md)
├── data/                         # Tracked split metadata & aggregate risk summary (data/README.md)
├── provenance/                   # Data, model, experiment & reproducibility docs (provenance/README.md)
├── docs/                         # Architecture, limitations & acquisition docs (docs/README.md)
├── artifacts/                    # Excluded binary artifact registry & SHA-256 hashes (artifacts/README.md)
├── results/                      # Sanitized aggregate tracking summary (results/README.md)
└── supplementary/                # 13 regenerable benchmark LaTeX tables & highlights (supplementary/README.md)
```

---

## 12. Software Requirements

Recorded directly from the verified Windows AMD64 test environment:

| Category | Verified Environment / Specification |
| :--- | :--- |
| **Python** | `3.11.9` (`requires-python >= 3.11` in `pyproject.toml`) |
| **Core Deep Learning (Required for inference/tests)** | `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `timm>=0.9.0`, `pytorch-lightning>=2.0.0` |
| **CUDA / Hardware** | Verified on CUDA `12.1` GPU (`torch.cuda.is_available() == True`); CPU-only execution supported for unit tests, aggregate tables/figures, and clean-room notebooks |
| **Scientific & Vision Core (Required)** | `numpy==2.4.6`, `scipy==1.17.1`, `scikit-learn==1.9.1`, `pandas==3.0.5`, `matplotlib==3.11.1`, `opencv-python==5.0.0.93`, `albumentations==1.3.1`, `Pillow==12.3.0`, `PyYAML==6.0.3`, `hydra-core==1.3.5`, `pydantic==2.13.4`, `pytest==9.1.1`, `loguru==0.7.3`, `tqdm==4.70.0` |
| **Optional Notebook Toolchain** | Pinned in [`requirements-notebooks.txt`](requirements-notebooks.txt): `jupyter==1.1.1`, `ipykernel==7.4.0`, `nbformat==5.11.1`, `nbclient==0.11.0`, `nbconvert==7.17.2`, `matplotlib==3.11.1`, `pandas==3.0.5` |
| **Optional / Excluded Heavy Dependencies** | `bitsandbytes>=0.46.1` (only for legacy local VLM experiments; not needed for DERMA-ACT V2 benchmark) |
| **Dataset & Artifact Requirements** | Raw ISIC 2019 (`CC-BY-NC 4.0`) imagery/CSVs and frozen `.ckpt`/`.pt`/`.npz` binaries are required **only** for `REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT` workflows |

---

## 13. Installation

Create an isolated Python 3.11 environment and install dependencies:

```bash
# 1. Create and activate virtual environment [NOT_VERIFIED — host-specific shell setup]
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate

# 2. Install core dependencies [NOT_VERIFIED in fresh venv; installed packages verified in Python 3.11.9 host]
pip install -r requirements.txt

# 3. Install pinned Jupyter notebook toolchain [VERIFIED]
pip install -r requirements-notebooks.txt

# 4. Verify environment and core test suite [VERIFIED]
python -m pytest -p no:cacheprovider -v tests/test_canonical_math.py tests/test_release_smoke.py
```

---

## 14. Dataset Acquisition and Preparation

Raw ISIC 2019 imagery and metadata CSVs are **not** bundled or redistributed in this repository (`CC-BY-NC 4.0`).

1. Download **ISIC 2019 Training Input**, **Training GroundTruth CSV**, and **Training Metadata CSV** directly from [https://challenge.isic-archive.com/data/#2019](https://challenge.isic-archive.com/data/#2019).
2. Place acquired files into the local ignored data directories:
   - `data/isic2019/raw/ISIC_2019_Training_Metadata.csv`
   - `data/isic2019/raw/ISIC_2019_Training_GroundTruth.csv`
   - `data/isic2019/ISIC_2019_Training_Input/*.jpg`
3. Verify partition census (`Train 16,702`, `Val 3,542`, `Cal 1,401`, `Test 3,686`; `split_hash = df52266ea5c2a4b1`) using [`notebooks/01_ISIC2019_Data_and_Split_Audit.ipynb`](notebooks/01_ISIC2019_Data_and_Split_Audit.ipynb).

See [data/README.md](data/README.md) and [docs/DATA_ACQUISITION_AND_ATTRIBUTION.md](docs/DATA_ACQUISITION_AND_ATTRIBUTION.md) for full layout details and mandatory upstream attributions (`BCN_20000`, `HAM10000`, `MSK`).

---

## 15. Quick Start

All commands below execute cleanly on CPU without raw ISIC 2019 imagery or binary checkpoints:

```bash
# 1. Bytecode compile check across active modules and scripts [VERIFIED]
python -m compileall -q src research/derma_act/core scripts tests

# 2. Run clean-room mathematical parity and repository hygiene smoke tests [VERIFIED]
python -m pytest -p no:cacheprovider -v tests/test_canonical_math.py tests/test_release_smoke.py

# 3. Regenerate all 13 benchmark LaTeX tables into an ignored output directory [VERIFIED]
python scripts/generate_tables.py --out-dir ./outputs/tables

# 4. Regenerate standalone aggregate architecture, risk-cost, and graphical abstract figures [VERIFIED]
python scripts/build_figures.py --tier aggregate --out-dir ./outputs/figures
```

---

## 16. Jupyter Notebook Tutorials

A curated seven-notebook research tutorial and audit suite resides in [`notebooks/`](notebooks/README.md). Public `.ipynb` files are committed with stripped outputs (`execution_count: null`) and portable repository-root discovery.

| Order | Notebook | Scope | Data / Artifact Requirement |
| :---: | :--- | :--- | :--- |
| 00 | [`00_Project_Overview_and_Quickstart.ipynb`](notebooks/00_Project_Overview_and_Quickstart.ipynb) | Environment audit, API smoke checks, config/script discovery | None (clean-room) |
| 01 | [`01_ISIC2019_Data_and_Split_Audit.ipynb`](notebooks/01_ISIC2019_Data_and_Split_Audit.ipynb) | Partition census, 7-class mapping, operational grouping, overlap audit | Tracked `split_metadata.json`; raw CSVs optional (`REQUIRES_DATA`) |
| 02 | [`02_Baseline_Classifier_Evaluation.ipynb`](notebooks/02_Baseline_Classifier_Evaluation.ipynb) | EfficientNet-B4 `A0` protocol, discrimination, confusion matrix, calibration | Canonical tables tracked; `test_canonical.npz` optional (`REQUIRES_MODEL_ARTIFACT`) |
| 03 | [`03_Sequential_View_Policy_Analysis.ipynb`](notebooks/03_Sequential_View_Policy_Analysis.ipynb) | `A0`–`A9` action mechanics, `A7`/`A9` flaws, V1 vs CAL-Stop & `Phase F` | Aggregate CSV/JSON tracked; `per_action_test.npz` & `policy.pt` optional (`REQUIRES_MODEL_ARTIFACT`) |
| 04 | [`04_Calibration_and_Conformal_Analysis.ipynb`](notebooks/04_Calibration_and_Conformal_Analysis.ipynb) | Temperature scaling, pooled & Mondrian LAC sets, `AKIEC` under-coverage | Aggregate CSV/JSON tracked; `calibration_canonical.npz` & `test_canonical.npz` optional (`REQUIRES_MODEL_ARTIFACT`) |
| 05 | [`05_Decision_Flip_and_Robustness_Analysis.ipynb`](notebooks/05_Decision_Flip_and_Robustness_Analysis.ipynb) | Pre-view instability ranking (`M = 14,815`) and `A0` perturbation bounds (`N = 50`) | Tracked aggregate JSON/tables (`tab5`, `tab7`) |
| 06 | [`06_Reproduce_Research_Tables_and_Figures.ipynb`](notebooks/06_Reproduce_Research_Tables_and_Figures.ipynb) | Safe regeneration of 13 LaTeX tables and aggregate figures + provenance verification | None for aggregate tier; raw ISIC images/caches for `--tier all` (`REQUIRES_DATA`) |

Launch interactively via `jupyter lab notebooks/` (`REQUIRES_OPTIONAL_DEPENDENCY` — install `requirements-notebooks.txt`).

---

## 17. Reproduction Commands

Every command below is explicitly classified according to its verified execution requirements:

| Command | Status Tag | Notes / Prerequisites |
| :--- | :---: | :--- |
| `python -m compileall -q src research/derma_act/core scripts tests` | `VERIFIED` | Runs cleanly in public clone and local workspace |
| `python -m pytest -p no:cacheprovider -v tests/test_canonical_math.py tests/test_release_smoke.py` | `VERIFIED` | 6/6 clean-room unit and hygiene checks pass |
| `python scripts/generate_tables.py --out-dir ./outputs/tables` | `VERIFIED` | Regenerates all 13 `.tex` tables from tracked Tier A artifacts |
| `python scripts/build_figures.py --tier aggregate --out-dir ./outputs/figures` | `VERIFIED` | Regenerates Figure 1, Figure 4, and Graphical Abstract |
| `jupyter lab notebooks/` | `REQUIRES_OPTIONAL_DEPENDENCY` | Requires `pip install -r requirements-notebooks.txt` |
| `python scripts/evaluate_v1_test.py` | `REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT` | Requires local `per_action_test.npz`, `per_action_dev.npz`, and `policy.pt` |
| `python scripts/run_p7_1_conformal_audit.py --root <DATA_ROOT>` | `REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT` | Requires `calibration_canonical.npz`, `test_canonical.npz`, ISIC metadata CSV |
| `python scripts/run_phase_f_restricted_policy.py --root <DATA_ROOT>` | `REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT` | Requires `per_action_test.npz`, `policy.pt`, ISIC metadata CSV |
| `python scripts/run_phase_g_mondrian_conformal.py --root <DATA_ROOT>` | `REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT` | Requires `calibration_canonical.npz`, `test_canonical.npz`, ISIC metadata CSV |
| `python scripts/build_figures.py --tier all --root <DATA_ROOT> --out-dir ./outputs/figures` | `REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT` | Full Figures 1–8 require raw ISIC `.jpg` images and `.npz` logit caches |
| `python -m src.training.train_pipeline` | `NOT_VERIFIED` / `REQUIRES_DATA` | Full end-to-end backbone retraining from scratch not re-executed in P8.12; historical pre-freeze optimization seeds are non-recoverable |

---

## 18. Evaluation and Statistical Analysis

- **Resampling Unit:** Cluster-stratified bootstrap (`B = 2,000`, `seed = 20261112`) resampling across the `K = 1,972` operational grouping units (`1,695` observed `lesion_id` clusters + `277` missing-ID singletons).
- **Paired Hypothesis Testing:** Exact two-sided McNemar tests on paired held-out test correctness indicators with Holm–Bonferroni family-wise error control:
  - Strategy benchmark (`tab3`): family size `m = 25`
  - Restricted spatial policy ablation (`tab9`): family size `m = 20`
  - Per-class sensitivity trade-offs (`tab4`): family size `m = 7`
- **Conformal Intervals:** Cluster-stratified bootstrap 95% CIs are reported for global and group-level coverage; pointwise image-level Wilson 95% CIs are reported alongside per-class Mondrian counts with explicit disclosure that Wilson intervals do not adjust for within-lesion correlation.

---

## 19. Figures and Tables Generation

- **Benchmark Tables (`13` LaTeX tables):** Generated deterministically by [`scripts/generate_tables.py`](scripts/generate_tables.py) from tracked aggregate CSV/JSON ledgers (`tab1`–`tab10`, `tab_s1`, `tab_s2`, `tab_s5`). Pre-generated copies reside in `supplementary/tables/` (public release) and are verified byte-consistent against canonical inputs.
- **Figures (`scripts/build_figures.py`):**
  - `--tier aggregate` (`VERIFIED`): runs `fig1_architecture.py`, `fig4_multiview_risk.py`, and `generate_graphical_abstract.py` cleanly without raw imagery.
  - `--tier all` (`REQUIRES_DATA`): runs Figures 1–8 (`scripts/figure_sources/fig1`–`fig8`), requiring local ISIC 2019 `.jpg` samples and frozen `.npz` caches.

---

## 20. Data Provenance and Availability

- **Canonical Data Governance:** See [data/README.md](data/README.md), [provenance/DATA_PROVENANCE.md](provenance/DATA_PROVENANCE.md), and [docs/DATA_ACQUISITION_AND_ATTRIBUTION.md](docs/DATA_ACQUISITION_AND_ATTRIBUTION.md).
- **Excluded Datasets:** Raw ISIC 2019 images (`CC-BY-NC 4.0`), synthetic smoke placeholders (`data/test_dataset*`), unverified standalone images, invalidated PAD-UFES-20 runs, and non-diagnostic HIBA latency logs are strictly excluded from public release and efficacy claims.

---

## 21. Model and Experiment Artifact Availability

- **Model & Weight Provenance:** See [artifacts/README.md](artifacts/README.md) and [provenance/MODEL_PROVENANCE.md](provenance/MODEL_PROVENANCE.md).
  - Frozen classifier `epoch=09-val_auroc=0.0000.ckpt` SHA-256: `0d14579d668d02781874cf91ad493fac4bf33ed53e3be928e3a7f9bb856ebbbb` (excluded binary; derived from `CC-BY-NC 4.0` data).
  - Frozen policy `policy.pt` (`6,787` params) SHA-256: `32a098c8e2aa958959122cb0292eff0cc8932ea4dac4c2599f3dd37ed2587c95` (excluded binary).
  - Frozen configs tracked directly in repository: `lambda_selection.json` (`3662d80e...`) and `stopping_threshold.json` (`dc13a238...`).
- **Experiment Provenance:** Tracked aggregate ledgers are documented in [provenance/EXPERIMENT_PROVENANCE.md](provenance/EXPERIMENT_PROVENANCE.md). Heavy `.ckpt`/`.npz` archival hosting (e.g., non-commercial Zenodo deposit) remains pending owner action.

---

## 22. License Status

> **No repository-wide open-source license has been established.**

Original author source code in this repository is currently published **without an open-source license grant** (all rights reserved until the author selects and commits an explicit `LICENSE` file). Upstream ISIC 2019 data and derivative weights remain governed by **`CC-BY-NC 4.0`**, and third-party Python packages remain under their respective upstream open-source licenses. See [`LICENSE_STATUS.md`](LICENSE_STATUS.md) for the full layer-by-layer governance table.

---

## 23. Citation

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{le2026derma_act_v2,
  author       = {Le, Duc-Son},
  title        = {DERMA-ACT V2: Event-Aware Sequential Virtual-View Policy Evaluation and Conformal Uncertainty Quantification for Dermoscopic Skin Lesion Classification},
  year         = {2026},
  version      = {0.2.0},
  url          = {https://github.com/noenc207/DERMA-ACT},
  orcid        = {0009-0000-5804-7811}
}
```

When using ISIC 2019 data, also cite **HAM10000** (`doi:10.1038/sdata.2018.161`), **MSK / ISIC 2019** (`arXiv:1710.05006`, `arXiv:1902.03368`), and **BCN20000** (`Hernandez-Perez et al., Sci. Data 2024`).

---

## 24. Ethics and Research-Only Disclaimer

- **Secondary Data Ethics:** This study exclusively involves secondary computational analysis of publicly archived, anonymized ISIC 2019 benchmark data; institutional review board (IRB) approval and informed consent were not required.
- **Clinical Safety Disclaimer:** DERMA-ACT V2 is experimental research software intended exclusively for methodological benchmarking and reproducibility analysis. It has **not** undergone clinical trials, regulatory clearance, external multi-center validation, or clinician reader studies, and must **never** be deployed for medical diagnosis or patient care.

---

## 25. Contact

- **Author & Maintainer:** Duc-Son Le
- **Academic Email:** `SonldQE220036@fpt.edu.vn`
- **ORCID:** [0009-0000-5804-7811](https://orcid.org/0009-0000-5804-7811)
- **GitHub Repository:** [https://github.com/noenc207/DERMA-ACT](https://github.com/noenc207/DERMA-ACT)
