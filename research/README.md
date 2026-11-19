# Research Implementation & Canonical Evidence (`research/`)

Houses the DERMA-ACT canonical mathematical specification (`research/derma_act/core/`), frozen policy configuration registry (`research/derma_act/config.py`), frozen scalar hyperparameter JSONs, and pure aggregate benchmark audit summaries.

## Relationship to `src/`

- **`src/`** provides modular neural architectures (`MultiHeadActionPolicy`), image transformation operators (`ViewGenerator`), and split-conformal primitives (`ConformalPredictor`).
- **`research/derma_act/`** pins the exact scientific evaluation equations, frozen constants (`STATE_DIM = 28`, `CANDIDATE_DIM = 9`, `N_CLASSES = 7`, `BUDGET = 5`, `LAMBDA = (5.0, 1.0, 0.5, 0.25)`, `TAU = -0.4286`, `EPS = 1e-12`), and verified aggregate result ledgers cited by the V3.2.1 manuscript and notebooks.

## What This Directory Contains

| Path | Role |
| :--- | :--- |
| `research/derma_act/config.py` | Single source of truth for dimensions, class order (`mel, nv, bcc, akiec, bkl, df, vasc`), action display labels, `load_lambda()`, `load_tau()`, and expected artifact SHA-256 digests |
| `research/derma_act/core/fusion.py` | `fuse_posterior(p_current, p_candidate, num_fused)` — recursive arithmetic running mean |
| `research/derma_act/core/state.py` | `entropy`, `top2_margin`, `construct_state` — canonical 28-D state vector |
| `research/derma_act/core/stopping.py` | `should_stop(max_utility, tau)` — utility-threshold stopping rule |
| `research/derma_act/core/transforms.py` | `get_original_transform()` (`A0`: Resize 291 -> CenterCrop 224) vs `get_view_transform()` (`A1`–`A9`: Normalize only) |
| `research/derma_act/runs/derma_act_v1/policy/` | Tracked `lambda_selection.json` and `stopping_threshold.json` (binary `policy.pt` excluded from public Git) |
| `research/derma_act/evidence/v1/` | Tracked aggregate `FINAL_RISK_COST_TABLE.csv` |
| `research/derma_act/p5_4_independent_policy_validation/` | `P5_4_BENCHMARK_ANALYSIS.json` (paired McNemar + Holm–Bonferroni `m = 25`) |
| `research/derma_act/p5_5_final_claim_closure/` | `decision_flip_cluster_results.json`, `minority_class_audit.json` |
| `research/derma_act/p5_6_action_quality_audit/` | `A7`/`A9` cohort summaries, action prediction metrics, registry, frequencies, transition matrices |
| `research/derma_act/p7_1_conformal_quantile_audit/` | `P7_1_GROUP_COVERAGE_RECONCILIATION.csv` |
| `research/derma_act/p8_1_final_submission_audit/` | `phase_f_restricted_policy/` and `phase_g_mondrian_conformal/` aggregate CSV/JSON outputs |

## What This Directory Does Not Contain (Public Release)

- Trained binary policy weights (`policy.pt`), per-image `.npz` logit caches (`per_action_test.npz`, `calibration_canonical.npz`, `test_canonical.npz`), local VLM prototypes, and session `.key` files are excluded from public Git distribution (see [artifacts/README.md](../artifacts/README.md) and [LICENSE_STATUS.md](../LICENSE_STATUS.md)).

## Verification Commands

```bash
# Verify canonical fusion, entropy, state, and transform parity [VERIFIED]
python -m pytest -p no:cacheprovider -v tests/test_canonical_math.py tests/test_release_smoke.py
```

## Related Documentation

- [Source Modules](../src/README.md) · [Experiment Provenance](../provenance/EXPERIMENT_PROVENANCE.md) · [Model Provenance](../provenance/MODEL_PROVENANCE.md) · [Limitations](../docs/LIMITATIONS.md)
