# Scripts Index (`scripts/`)

Reproducible command-line runners for benchmark table/figure generation, data-dependent statistical audits, and (in the local research workspace) historical analysis, cohort packaging, and verification helpers.

## Categorized Script Index

### 1. Plotting & Table Generation (`VERIFIED` aggregate tier)
| Script | Purpose | Requirement Tag | Tested Command |
| :--- | :--- | :---: | :--- |
| `scripts/generate_tables.py` | Regenerates all 13 benchmark LaTeX tables (`tab1`–`tab10`, `tab_s1`, `tab_s2`, `tab_s5`) from tracked aggregate JSON/CSV ledgers | `VERIFIED` | `python scripts/generate_tables.py --out-dir ./outputs/tables` |
| `scripts/build_figures.py` | Orchestrates standalone aggregate (`--tier aggregate`) or full data-dependent (`--tier all`) figure generation | `VERIFIED` (`aggregate`) / `REQUIRES_DATA` (`all`) | `python scripts/build_figures.py --tier aggregate --out-dir ./outputs/figures` |
| `scripts/generate_graphical_abstract.py` | Generates standalone Graphical Abstract assets | `VERIFIED` | Invoked by `build_figures.py --tier aggregate` |
| `scripts/figure_sources/fig1`–`fig8` | Individual publication figure scripts (`fig1_architecture.py` .. `fig8_perturbation_robustness.py`) | `VERIFIED` (`fig1`, `fig4`) / `REQUIRES_DATA` (`fig2`, `fig3`, `fig5`–`fig8`) | Invoked via `scripts/build_figures.py` |

### 2. Benchmarking & Statistical Audits (`REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT`)
| Script | Purpose | Requirement Tag | Command |
| :--- | :--- | :---: | :--- |
| `scripts/evaluate_v1_test.py` | Evaluates Learned Policy V1 on frozen per-action test logits | `REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT` | `python scripts/evaluate_v1_test.py` |
| `scripts/run_p7_1_conformal_audit.py` | Recomputes split-conformal quantiles and multi-estimand group coverage | `REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT` | `python scripts/run_p7_1_conformal_audit.py --root <DATA_ROOT>` |
| `scripts/run_phase_f_restricted_policy.py` | Runs post-hoc 5-spatial-action candidate masking ablation (`{A1,A3,A4,A5,A6}`) | `REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT` | `python scripts/run_phase_f_restricted_policy.py --root <DATA_ROOT>` |
| `scripts/run_phase_g_mondrian_conformal.py` | Runs image-level Mondrian class-conditional LAC calibration & comparison | `REQUIRES_DATA` / `REQUIRES_MODEL_ARTIFACT` | `python scripts/run_phase_g_mondrian_conformal.py --root <DATA_ROOT>` |

### 3. Organized Subfolders (Local Research Workspace)
Subdirectories below hold additional relocated internal workspace scripts (documented individually; only portable clean-room scripts above are populated in the public repository):
- [`scripts/entrypoints/README.md`](entrypoints/README.md) — Local demo launcher (`RUN_DEMO.ps1`) and pilot runner (`run_pilot.py`).
- [`scripts/data/README.md`](data/README.md) — Local case construction, sampling, and UI packaging scripts.
- [`scripts/analysis/README.md`](analysis/README.md) — Local stratified bootstrap, sensitivity, and phase analysis scripts.
- [`scripts/audit/README.md`](audit/README.md) — Local cohort, identity, claim, and phase integrity audit scripts.
- [`scripts/plotting/README.md`](plotting/README.md) — Pointer to clean-room figure orchestrator and `figure_sources/`.
- [`scripts/utilities/README.md`](utilities/README.md) — Manifest, prompt-freezing, and inventory helper scripts.

## What This Directory Does Not Contain

- Model training checkpoints, raw datasets, or generated outputs (always write outputs to an ignored directory such as `./outputs/`).

## Dependencies & Related Documentation

- Requires core Python dependencies (`requirements.txt`), especially `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, and `scikit-learn`.
- See [Reproducibility Guide](../provenance/REPRODUCIBILITY.md) · [Notebook 06](../notebooks/06_Reproduce_Research_Tables_and_Figures.ipynb) · [Tests](../tests/README.md).
