# Reproducibility Guide (`DERMA-ACT V2`)

## 1. Verified Execution Environment

Recorded directly in the local Windows AMD64 workspace during `P8.10` / `P8.11` verification:

- **Python:** `3.11.9` (`>=3.11`)
- **Key Installed Package Versions:**
  - `torch`: `2.5.1+cu121` (CUDA available locally; CPU supported for unit tests and aggregate scripts)
  - `numpy`: `2.4.6`
  - `pandas`: `3.0.5`
  - `scipy`: `1.17.1`
  - `scikit-learn`: `1.9.1`
  - `matplotlib`: `3.11.1`
  - `Pillow`: `12.3.0`
  - `opencv-python`: `5.0.0.93`
  - `albumentations`: `1.3.1`
  - `PyYAML`: `6.0.3`
  - `pydantic`: `2.13.4`
  - `pytest`: `9.1.1`

## 2. Established Seeds & Frozen Evaluation Artifacts

- **ISIC 2019 Partition Seed (`data/isic2019/splits/split_metadata.json`):** `seed = 42` (`split_hash = df52266ea5c2a4b1`).
- **Cluster-Stratified Bootstrap & Conformal Audits:** deterministic seeds (`20261112`) operating over frozen `.npz` logits.
- **Frozen Utility & Stopping Parameters (`research/derma_act/config.py`):** `LAMBDA_Q = 5.0`, `LAMBDA_C = 1.0`, `LAMBDA_I = 0.5`, `LAMBDA_K = 0.25`, `TAU = -3/7` (`-0.4286`), `EPS = 1e-12`, `BUDGET = 5`.
- **Training vs. Downstream Inference Reproducibility Distinction:** Neither the historical pre-freeze EfficientNet-B4 optimization seeds nor unseeded PyTorch weight initialization / DataLoader shuffling seeds in the offline policy trainer are recoverable for exact bit-for-bit retraining from scratch. Frozen `.ckpt`, `policy.pt`, and cached `.npz` evaluation logits govern deterministic downstream inference reproducibility.

## 3. Standalone Commands (`VERIFIED` without raw ISIC images)

```bash
# 1. Bytecode syntax compile check across V2 Python packages and scripts [VERIFIED]
python -m compileall -q src research/derma_act/core scripts tests

# 2. Core DERMA-ACT unit math parity and hygiene smoke tests [VERIFIED]
python -m pytest -p no:cacheprovider -v tests/test_canonical_math.py tests/test_release_smoke.py

# 3. Regenerate manuscript LaTeX tables from aggregate audit JSON/CSV ledgers [VERIFIED]
python scripts/generate_tables.py --out-dir ./outputs/tables

# 4. Regenerate standalone vector Figure 1, Figure 4, and Graphical Abstract [VERIFIED]
python scripts/build_figures.py --tier aggregate --out-dir ./outputs/figures
```

## 4. Data-Dependent Evaluation Commands (`REQUIRES_DATA`)

Executing full test-set evaluation, conformal quantile derivation from per-image logits, or Figures 2, 3, 5, 6, 7, and 8 requires acquiring upstream **`CC-BY-NC 4.0`** ISIC 2019 imagery and placing local `.ckpt`/`.pt`/`.npz` artifacts outside Git tracking (protected by `.gitignore`):

| Script / Command | Requirement Status | Required Local Artifacts |
| :--- | :---: | :--- |
| `python scripts/evaluate_v1_test.py` | `REQUIRES_DATA` | `per_action_test.npz`, `per_action_dev.npz`, `policy.pt` |
| `python scripts/run_p7_1_conformal_audit.py --root <DATA_ROOT>` | `REQUIRES_DATA` | `calibration_canonical.npz`, `test_canonical.npz`, ISIC metadata CSV |
| `python scripts/run_phase_f_restricted_policy.py --root <DATA_ROOT>` | `REQUIRES_DATA` | `per_action_test.npz`, `policy.pt` |
| `python scripts/run_phase_g_mondrian_conformal.py --root <DATA_ROOT>` | `REQUIRES_DATA` | `calibration_canonical.npz`, `test_canonical.npz` |
| `python scripts/build_figures.py --tier all --root <DATA_ROOT>` | `REQUIRES_DATA` | Raw ISIC `.jpg` images (e.g., `ISIC_0012970.jpg`) and cached `.npz` logits |
| `pytest tests/test_golden_parity_regression.py tests/test_demo_*.py` | `REQUIRES_DATA` / local `live` + checkpoints | Requires untracked `research/derma_act/live/` and frozen weights (verified `PASS` in local workspace) |
