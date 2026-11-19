# Documentation Index (`docs/`)

Central navigation hub for DERMA-ACT V2 architectural specifications, methodological limitations, dataset governance, reproducibility tiers, and developer reference notes.

## Core Authoritative Documents

| Document | Purpose |
| :--- | :--- |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Bounded technical specification of EfficientNet-B4 baseline, `A0`–`A9` virtual views, 28-D state, recursive arithmetic fusion, `MultiHeadActionPolicy` (`6,787` params), and conformal calibration |
| [`LIMITATIONS.md`](LIMITATIONS.md) | Comprehensive registry of study boundaries (no `patient_id`, supervised surrogate vs POMDP, arithmetic mean vs Bayesian, `A7`/`A9` flaws, `CAL-Stop` parity, `AKIEC` conformal gap, no external/reader validation) |
| [`DATA_ACQUISITION_AND_ATTRIBUTION.md`](DATA_ACQUISITION_AND_ATTRIBUTION.md) | Official ISIC 2019 acquisition instructions, mandatory `CC-BY-NC 4.0` copyright attributions (`BCN_20000`, `HAM10000`, `MSK`), and local artifact placement |

## Thematic Navigation Directories

| Subdirectory | Navigation Entry Point |
| :--- | :--- |
| `docs/architecture/` | [`architecture/README.md`](architecture/README.md) — System architecture pointers |
| `docs/methodology/` | [`methodology/README.md`](methodology/README.md) — Experimental methodology & limitations pointers |
| `docs/reproducibility/` | [`reproducibility/README.md`](reproducibility/README.md) — Reproduction tiers & notebook suite pointers |
| `docs/data-governance/` | [`data-governance/README.md`](data-governance/README.md) — Dataset licensing & legal separation pointers |
| `docs/development/` | [`development/README.md`](development/README.md) — Developer conventions & local notes |

## Related Entry Points

- [Root README](../README.md) · [Provenance Index](../provenance/README.md) · [Jupyter Notebooks](../notebooks/README.md) · [License Status](../LICENSE_STATUS.md)
