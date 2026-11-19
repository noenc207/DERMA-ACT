# DERMA-ACT V2 Jupyter Notebook Suite (`notebooks/`)

Curated, executable Python 3 Jupyter notebooks designed for transparent methodological exploration, statistical auditing, and clean-room reproduction.

## Notebook Index & Verified Execution Matrix

| Notebook | Purpose | Data requirement | Execution status (Public Clean-Room) | Execution status (Local Full-Data Workspace) |
| :--- | :--- | :--- | :---: | :---: |
| [`00_Project_Overview_and_Quickstart.ipynb`](00_Project_Overview_and_Quickstart.ipynb) | Project overview, environment/package verification, YAML config validation, and canonical API smoke check | None (standalone clean-room) | `FULL_EXECUTION_PASS` | `FULL_EXECUTION_PASS` |
| [`01_ISIC2019_Data_and_Split_Audit.ipynb`](01_ISIC2019_Data_and_Split_Audit.ipynb) | ISIC 2019 provenance, 7-class mapping (`AK`+`SCC`->`AKIEC`), frozen split census, operational grouping, and overlap audit | Tracked `split_metadata.json`; raw ISIC CSVs optional (`REQUIRES_DATA`) | `PARTIAL_EXECUTION_DATA_REQUIRED` | `FULL_EXECUTION_PASS` |
| [`02_Baseline_Classifier_Evaluation.ipynb`](02_Baseline_Classifier_Evaluation.ipynb) | Frozen EfficientNet-B4 `A0` protocol, per-class operating characteristics, temperature scaling, and 7x7 confusion matrix | Canonical tables tracked; `test_canonical.npz` & `calibration_canonical.npz` optional (`REQUIRES_MODEL_ARTIFACT`) | `PARTIAL_EXECUTION_DATA_REQUIRED` | `FULL_EXECUTION_PASS` |
| [`03_Sequential_View_Policy_Analysis.ipynb`](03_Sequential_View_Policy_Analysis.ipynb) | `A0`–`A9` action mechanics, `A7`/`A9` operator audits, Learned V1 vs `CAL-Stop`, and `Phase F` 5-spatial-action ablation | Aggregate CSV/JSON tracked; local sample JPEG, `policy.pt`, `per_action_test.npz` optional (`REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT`) | `PARTIAL_EXECUTION_DATA_REQUIRED` | `FULL_EXECUTION_PASS` |
| [`04_Calibration_and_Conformal_Analysis.ipynb`](04_Calibration_and_Conformal_Analysis.ipynb) | Temperature scaling, pooled vs image-level Mondrian LAC, multi-estimand coverage, and small-sample `AKIEC` analysis | Aggregate P7.1 & Phase G CSV/JSON tracked; `calibration_canonical.npz` & `test_canonical.npz` optional (`REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT`) | `PARTIAL_EXECUTION_DATA_REQUIRED` | `FULL_EXECUTION_PASS` |
| [`05_Decision_Flip_and_Robustness_Analysis.ipynb`](05_Decision_Flip_and_Robustness_Analysis.ipynb) | Pre-acquisition decision-flip ranking (`M=14,815`) reliability curve and `A0` digital perturbation sensitivity (`N=50`) | None beyond tracked aggregate JSON & tables (`tab5`, `tab7`) | `FULL_EXECUTION_PASS` | `FULL_EXECUTION_PASS` |
| [`06_Reproduce_Research_Tables_and_Figures.ipynb`](06_Reproduce_Research_Tables_and_Figures.ipynb) | Safe regeneration of 13 benchmark LaTeX tables and standalone aggregate figures (`Fig 1`, `Fig 4`, Graphical Abstract) + SHA-256 verification | None for aggregate tier; raw ISIC images/caches for `--tier all` (`REQUIRES_DATA`) | `FULL_EXECUTION_PASS` | `FULL_EXECUTION_PASS` |

## Recommended Execution Order

Run notebooks sequentially from **`00`** through **`06`**:
1. Start with `00` to verify Python `3.11+`, package versions, and repository-root discovery.
2. Run `01` to inspect ISIC 2019 partition counts (`16,702 / 3,542 / 1,401 / 3,686`) and operational grouping boundaries.
3. Run `02` → `03` → `04` → `05` to examine baseline performance, sequential virtual-view policies, conformal calibration, and instability/perturbation audits.
4. Run `06` to regenerate all `13` benchmark LaTeX tables and aggregate figures into `outputs/notebooks/nb06_reproduction/` without altering frozen files.

## Setup & Hygiene Policy

- **Install Notebook Dependencies (`VERIFIED`):** `pip install -r requirements-notebooks.txt` (pinned `jupyter==1.1.1`, `ipykernel==7.4.0`, `nbformat==5.11.1`, `nbclient==0.11.0`, `nbconvert==7.17.2`, `matplotlib==3.11.1`, `pandas==3.0.5`).
- **Launch (`REQUIRES_OPTIONAL_DEPENDENCY`):** `jupyter lab notebooks/`
- **Clean Git Hygiene:** All committed `.ipynb` files are stripped of execution outputs (`execution_count: null`, zero embedded binary blobs/workstation paths). Generated figures and tables are written exclusively to gitignored `outputs/notebooks/`.
- **Honest Data Preflight:** Notebooks `01`–`04` explicitly check for restricted `CC-BY-NC 4.0` ISIC 2019 CSVs/images and excluded `.npz`/`.ckpt`/`.pt` artifacts. Missing local files trigger explicit `SKIPPED` notices rather than fabricated data.

## Related Documentation

- [Root README](../README.md) · [Data Acquisition](../data/README.md) · [Artifacts Registry](../artifacts/README.md) · [Reproducibility Guide](../provenance/REPRODUCIBILITY.md)
