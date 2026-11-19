# Model Checkpoints & Cached Tensor Artifacts (`artifacts/`)

Registry documenting which DERMA-ACT V2 artifacts are tracked directly in Git, which heavy binary artifacts are excluded from public distribution, and how authorized owners verify SHA-256 integrity or regenerate permitted aggregate outputs.

## What Is Distributed vs. Excluded

| Artifact Category | Status in Public Git | Reason / Governance |
| :--- | :---: | :--- |
| Frozen scalar policy configs (`lambda_selection.json`, `stopping_threshold.json`) | **Included** | Lightweight JSON hyperparameters under `research/derma_act/runs/derma_act_v1/policy/` |
| Pure aggregate benchmark CSV/JSON summaries (`research/derma_act/`, `data/aggregate/`) | **Included** | Zero image pixels, zero personal/patient IDs; powers tables, aggregate figures, and notebooks |
| Regenerable LaTeX benchmark tables (`supplementary/tables/*.tex`) | **Included** | Deterministic output of `scripts/generate_tables.py` |
| EfficientNet-B4 weights (`epoch=09-val_auroc=0.0000.ckpt`) & `policy.pt` | **Excluded** | Binary size & derivative `CC-BY-NC 4.0` non-commercial governance (`LICENSE_STATUS.md`) |
| Per-image `.npz` logit caches (`calibration_canonical.npz`, `test_canonical.npz`, `per_action_test.npz`, `per_action_dev.npz`) | **Excluded** | Large binary caches (>3.2 GiB total locally) derived from ISIC 2019 imagery |

## Expected SHA-256 Verification Digests (`research/derma_act/config.py`)

| Artifact | Role | Expected SHA-256 |
| :--- | :--- | :--- |
| `epoch=09-val_auroc=0.0000.ckpt` | Frozen EfficientNet-B4 backbone weights (`17.9M` params) | `0d14579d668d02781874cf91ad493fac4bf33ed53e3be928e3a7f9bb856ebbbb` |
| `policy.pt` | Frozen `MultiHeadActionPolicy` (`6,787` params) | `32a098c8e2aa958959122cb0292eff0cc8932ea4dac4c2599f3dd37ed2587c95` |
| `lambda_selection.json` | Utility coefficients `(5.0, 1.0, 0.5, 0.25)` | `3662d80eacb9ca39276697ff5f9d043d3774794a9589f316b81b2c0fdf0dea05` |
| `stopping_threshold.json` | Stopping threshold `tau = -3/7` (`-0.4285714...`) | `dc13a238e16a678bf153beb8b2cbbb7341f8360f8e8d0f8172ee68bf67eaf8ef` |

## Regenerating Permitted Outputs Locally (`VERIFIED`)

```bash
# Regenerate all 13 benchmark LaTeX tables into ./outputs/tables
python scripts/generate_tables.py --out-dir ./outputs/tables

# Regenerate Figure 1, Figure 4, and Graphical Abstract into ./outputs/figures
python scripts/build_figures.py --tier aggregate --out-dir ./outputs/figures
```

## Related Documentation

- [Model Provenance](../provenance/MODEL_PROVENANCE.md) · [Experiment Provenance](../provenance/EXPERIMENT_PROVENANCE.md) · [License Status](../LICENSE_STATUS.md) · [Notebook 06](../notebooks/06_Reproduce_Research_Tables_and_Figures.ipynb)
