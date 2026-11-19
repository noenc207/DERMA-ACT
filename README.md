# DERMA-ACT V2 — Trustworthy Skin Cancer AI

A research framework for seven-class dermoscopic image classification and event-aware sequential image-view policy evaluation, with an emphasis on uncertainty quantification, operational constraints, calibration, and transparent evaluation.

> **Research Software Only:** This repository accompanies a retrospective computational comparative study and is **not** a clinical diagnostic product. No repository-wide open-source license has been established; see [LICENSE_STATUS.md](LICENSE_STATUS.md).

---

## 1. Project Status

- **Software Status:** Experimental research software / computational evaluation prototype.
- **Manuscript Status:** Pre-submission manuscript revision **V3.2.1** (*Biomedical Signal Processing and Control* target; not yet peer-reviewed or published). Unpublished manuscript sources, drafts, and internal submission documents are intentionally excluded from this public code release.
- **Clinical Authorization:** **None.** No FDA, CE, or institutional clinical deployment clearance exists; outputs must **not** be used for medical diagnosis or patient treatment decisions.

---

## 2. Research Scope and Objectives

DERMA-ACT V2 audits sequential virtual-view perception alongside finite-sample conformal uncertainty tools across six axes on the seven-class ISIC 2019 benchmark (`MEL`, `NV`, `BCC`, `AKIEC`, `BKL`, `DF`, `VASC`):

1. **Frozen Backbone Classification (`A0` Global View):** Multi-class discrimination using a frozen `EfficientNet-B4` backbone (`17.9M` parameters) and post-hoc temperature scaling ($T_{\mathrm{full}} = 0.8518$).
2. **Sequential Virtual-View Policy vs. Fixed Comparators:** Comparing `MultiHeadActionPolicy` (`A1`–`A9`, recursive arithmetic probability fusion up to $B=5$) against a validation-ranked fixed spatial sequence (`CAL-Stop`: `A6 -> A5 -> A3 -> A1 -> A4`).
3. **Restricted-Inventory Ablation (`Phase F`):** Post-hoc candidate masking of Learned Policy V1 to `{A1, A3, A4, A5, A6}` without policy retraining.
4. **Code-Level Operator Forensics:** Quantifying `A7` Gabor `np.uint8` saturation (`99.97%` median saturation; `25.69%` zero-collapse on validation) and `A9` morphological dark-structure masking (`11.55%` mean area).
5. **Pre-Acquisition Decision-Flip Instability Head:** Evaluating threshold-free ranking (`AUROC`, step-function `AP`) over $M = 14,815$ realized transitions (`988` flips, `6.67%` base rate).
6. **Split-Conformal and Mondrian Uncertainty (`P7.1` & `Phase G`):** Auditing finite-sample split-conformal sets ($\alpha = 0.10$, `k* = 634`, `q* = 0.904267`) alongside post-hoc image-level Mondrian calibration disparities (`AKIEC`).

---

## 3. Dataset Acquisition and Attribution

- **Source:** **ISIC 2019 Challenge Training Set** (https://challenge.isic-archive.com/data/#2019), governed by Creative Commons Attribution-NonCommercial 4.0 International (**`CC-BY-NC 4.0`**) incorporating `BCN_20000`, `HAM10000`, and `MSK`. Mandatory attributions are preserved in [docs/DATA_ACQUISITION_AND_ATTRIBUTION.md](docs/DATA_ACQUISITION_AND_ATTRIBUTION.md) and [provenance/DATA_PROVENANCE.md](provenance/DATA_PROVENANCE.md).
- **Non-Redistribution:** Raw `.jpg` imagery and row-level clinical metadata CSVs are **not** redistributed here.
- **Partition Sizes ([data/isic2019/splits/split_metadata.json](data/isic2019/splits/split_metadata.json), `seed = 42`):**

| Split | Images | Operational Groups (`K`) | Purpose |
| :--- | ---: | ---: | :--- |
| **Train** | `16,702` | `9,198` | Backbone training & offline policy prefix generation |
| **Validation** | `3,542` | `1,972` | Utility weight selection ($\lambda$) & `N = 50` single-view perturbation probe |
| **Calibration** | `1,401` | `789` | Global $T_{\mathrm{full}}$; split into `D_fit` (`698`) & `D_conf` (`703`) for LAC conformal |
| **Test** | `3,686` | `1,972` | Frozen single-split held-out evaluation benchmark |
| **Total** | **`25,331`** | **`13,931`** | Single-benchmark partition (`lesion_id` + singleton operational IDs) |

- **Patient-Level Independence Limitation:** ISIC 2019 metadata omits patient identifiers. **Patient-level independence is not established**, and cluster-stratified intervals reflect operational `lesion_id` plus singleton grouping only.

---

## 4. Verified Benchmark Results (`N = 3,686` ISIC 2019 Test Split)

All values below are traced directly to frozen aggregate artifacts in `research/derma_act/` ([provenance/EXPERIMENT_PROVENANCE.md](provenance/EXPERIMENT_PROVENANCE.md)):

| Evaluation Setting | Reported Metric | Value | Views / Support | Statistical & Audit Context | Source |
| :--- | :--- | ---: | :--- | :--- | :--- |
| **`A0` Global Baseline** | Top-1 Accuracy / Error | **69.21%** / **30.79%** | `1.00` view | Balanced Acc `68.83%`, Macro AUROC `0.9274`, ECE `2.71%` ($T=0.8518$) | `P5_4_BENCHMARK_ANALYSIS.json` |
| **Learned Policy V1** ($B=5$, 9-action) | Top-1 Accuracy / Error | **73.06%** / **26.94%** | `4.02` mean views | Balanced Acc `70.04%`; paired Holm `p_adj = 0.3185` (`0.319`) vs `CAL-Stop` | `P5_4_BENCHMARK_ANALYSIS.json` |
| **`CAL-Stop` Fixed Spatial** ($B=5$, 5-action) | Top-1 Accuracy / Error | **74.04%** / **25.96%** | `3.90` mean views | Validation-ranked (`A6->A5->A3->A1->A4`); no superiority over Learned V1 established | `P5_4_BENCHMARK_ANALYSIS.json` |
| **Restricted 5-Spatial-Action V1** | Top-1 Error / Accuracy | **26.21%** / **73.79%** | **`3.44`** mean views | Post-hoc masked without retraining; Holm **`p_adj = 1.000`** vs `CAL-Stop` | `phase_f_feasibility_and_results.json` |
| **Decision-Flip Instability Head** | AUROC / AP / Brier | **0.9473** / **0.5172** / **0.0416** | `M = 14,815` steps | Ranks `988` flips (`6.67%`); threshold-free score without validated cutoff | `decision_flip_cluster_results.json` |
| **Pooled Split-Conformal LAC** ($\alpha=0.10$) | Marginal Coverage | **90.67%** (`1.92` mean size) | `D_conf` $N=703, k^*=634$ | `q* = 0.904267`; minority `AKIEC` empirical coverage is **`74.21%`** | `P7_1_GROUP_COVERAGE_RECONCILIATION.csv` |
| **Image-Level Mondrian LAC** (`AKIEC`) | Empirical Coverage | **70.59%** (`156/221`) | `n_c = 30`, `K_c = 17` cal | Mean set size `2.39`; small-sample empirical behavior, **not** a 90% guarantee | `phase_g_feasibility_and_results.json` |

---

## 5. Methodology Summary

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [provenance/MODEL_PROVENANCE.md](provenance/MODEL_PROVENANCE.md):

- **Transform Operators:** `A0` applies `Resize(291) -> CenterCrop(224) -> ImageNet normalization`; `A1`–`A9` operate on unnormalized images resized to `224 x 224` (`research/derma_act/core/transforms.py`).
- **Probability Fusion:** Equal-weight recursive arithmetic running mean (`p_{t+1} = (b * p_t + p_cand)/(b + 1)` in `research/derma_act/core/fusion.py`) — strictly arithmetic averaging, **not** Bayesian posterior updating.
- **Surrogate Utility Policy:** `6,787`-parameter MLP (`28 + 9 -> 64 -> 64 -> 3 heads`) acting as a finite-horizon supervised surrogate value estimator (not a formally Bellman-optimized POMDP) with frozen weights `(5.0, 1.0, 0.5, 0.25)` and threshold `tau = -3/7` (`-0.4286`).

---

## 6. Repository Structure

```text
DERMA-ACT/
├── README.md                  # Public repository overview & benchmark guide
├── LICENSE_STATUS.md          # Asset-layer license separation notice
├── CITATION.cff               # Citation metadata (Duc-Son Le, ORCID 0009-0000-5804-7811)
├── pyproject.toml             # Python 3.11+ package configuration
├── requirements.txt           # Core runtime dependencies
├── requirements-lock.txt      # Pinned environment specification
├── .env.example               # Safe placeholder environment variables
├── .github/workflows/ci.yml   # Python 3.11 syntax, unit & aggregate reproduction CI
├── configs/                   # Hydra/YAML experiment and preprocessing configs
├── src/                       # Core modules (active perception & split-conformal)
├── research/derma_act/        # Core math, frozen policy configs & aggregate summaries
├── scripts/                   # Table/figure generators & test evaluation scripts
├── tests/                     # Standalone unit math & hygiene smoke tests
├── data/                      # Governance README, split metadata & aggregate CSV
├── docs/                      # Architecture, limitations & ISIC attribution docs
├── provenance/                # Data, model, experiment & reproducibility provenance
├── artifacts/                 # SHA-256 verification ledger for excluded checkpoints
├── results/                   # Sanitized aggregate tracking summary
└── supplementary/             # Regenerable LaTeX benchmark tables & highlights
```

---

## 7. Environment and Reproduction

Tested locally on `Python 3.11.9` (`requires-python >= 3.11` in [pyproject.toml](pyproject.toml)).

```bash
pip install -r requirements.txt
pip install pytest matplotlib   # required for test suite and Tier A figure generation
```

| Stage | Command | Requirement Status | Notes |
| :--- | :--- | :---: | :--- |
| **Syntax Check** | `python -m compileall -q src research/derma_act/core scripts tests` | `VERIFIED` | Standalone Python syntax validation |
| **Core Unit & Hygiene Tests** | `python -m pytest -p no:cacheprovider tests/test_canonical_math.py tests/test_release_smoke.py -v` | `VERIFIED` | Verifies transforms, fusion, stopping & hygiene (`6 passed`) |
| **LaTeX Table Generation** | `python scripts/generate_tables.py --out-dir ./outputs/tables` | `VERIFIED` | Rebuilds aggregate tables from `research/derma_act/` JSON/CSVs |
| **Aggregate Figure Generation** | `python scripts/build_figures.py --tier aggregate --out-dir ./outputs/figures` | `VERIFIED` | Rebuilds Figure 1, Figure 4, and Graphical Abstract |
| **Full Test Evaluation** | `python scripts/evaluate_v1_test.py` | `REQUIRES_DATA` | Requires local ISIC 2019 acquisition & excluded `.npz`/`.pt` files |
| **Conformal & Restricted Audits** | `python scripts/run_p7_1_conformal_audit.py` / `scripts/run_phase_f_restricted_policy.py` | `REQUIRES_DATA` | Requires local excluded `.npz` caches |

See [provenance/REPRODUCIBILITY.md](provenance/REPRODUCIBILITY.md) and [artifacts/README.md](artifacts/README.md) for artifact SHA-256 verification digests and boundary notes.

---

## 8. Known Limitations

1. **No Patient-Level Independence:** Patient identifiers are absent in ISIC 2019; grouping controls known `lesion_id` duplication only.
2. **Single-Benchmark Retrospective Scope:** Zero prospective trials, clinician reader studies, or validated external clinical cohorts were evaluated.
3. **No Proven Accuracy Superiority Over Fixed Sequence:** Learned Policy V1 (`73.06%`) does not outperform validation-ranked `CAL-Stop` (`74.04%`, `p_adj = 0.3185`) or show statistical separation in the masked 5-spatial-action ablation (`26.21%` error, `p_adj = 1.000`).
4. **Operator Artifacts & Calibration Boundaries:** `A7` Gabor filtering undergoes `np.uint8` saturation clipping (`99.97%`); `A9` performs non-selective dark-structure masking (`11.55%`); `AKIEC` conformal coverage (`74.21%` pooled, `70.59%` Mondrian under `n_c = 30, K_c = 17`) and sparse `DF`/`VASC` strata reflect small-sample limits.

---

## 9. License and Attribution

**No repository-wide open-source license has been established.** See [LICENSE_STATUS.md](LICENSE_STATUS.md) for the explicit separation between original author code (unlicensed pending explicit owner selection), third-party dependencies, and `CC-BY-NC 4.0` upstream ISIC 2019 data and derivatives.

---

## 10. Citation and Contact

See [CITATION.cff](CITATION.cff).

- **Corresponding Author:** Duc-Son Le
- **Affiliation:** FPT University, Quy Nhon Campus, An Phu Thinh New Urban Area, Quy Nhon Dong Ward, Gia Lai Province, Vietnam
- **Email:** [SonldQE220036@fpt.edu.vn](mailto:SonldQE220036@fpt.edu.vn)
- **ORCID:** [https://orcid.org/0009-0000-5804-7811](https://orcid.org/0009-0000-5804-7811)
