# Test Suite (`tests/`)

Automated verification tests covering canonical DERMA-ACT mathematics, frozen policy configuration integrity, repository hygiene, and (in the local workspace) regression and phase-audit suites.

## Test Groups & Execution Matrix

### 1. Clean-Room Unit & Hygiene Tests (Included in Public Release — `VERIFIED`)
| Test Module | Scope | Data / Checkpoint Requirement | Command & Verified Result |
| :--- | :--- | :---: | :--- |
| `tests/test_canonical_math.py` | Verifies Shannon entropy (`EPS = 1e-12`), recursive arithmetic fusion parity (`fuse_posterior`), `A0` vs `A1`–`A9` transform parity, and RGB channel integrity | None | `python -m pytest -p no:cacheprovider -v tests/test_canonical_math.py` (**4 passed**) |
| `tests/test_release_smoke.py` | Verifies frozen dimensions/hyperparameters, `MultiHeadActionPolicy` (`6,787` params) forward pass, and zero credentials / workstation paths / forbidden binary artifacts | None | `python -m pytest -p no:cacheprovider -v tests/test_release_smoke.py` (**2 passed**) |

Run both clean-room suites together:
```bash
python -m pytest -p no:cacheprovider -v tests/test_canonical_math.py tests/test_release_smoke.py
```

### 2. Local Workspace Integration, Parity & Statistical Audit Tests (Private Workspace Only)
In `D:` local development only (excluded from public release because they inspect local checkpoints, phase-2/3 artifacts, or legacy VLM modules):
- **Reorganized Core & Phase Audits (`VERIFIED` locally — 33 passed across 7 modules):** `test_canonical_math.py`, `test_release_smoke.py`, `test_golden_parity_regression.py` (`REQUIRES_MODEL_ARTIFACT`), `test_v2_phase2_grounding.py`, `test_v2_phase3_execution.py`, `test_v2_phase3_evaluation_harness.py`, `test_v2_final_statistical_audit.py`.
- **Demo Smoke Tests (`REQUIRES_MODEL_ARTIFACT`):** `test_demo_*.py` (require local `checkpoints/` and `research/derma_act/live/`).
- **Legacy Exploratory VLM & `repair1` Tests (`REQUIRES_OPTIONAL_DEPENDENCY` / historical prototype drift):** `test_v2_real_vlm_*.py` require optional `bitsandbytes>=0.46.1` (skipped when absent); historical `test_v2_vlm_*` / `test_v2_repair1_*` reflect pre-freeze exploratory multimodal prototypes outside the DERMA-ACT V2 manuscript scope.

## Fixtures & Conventions

- `pytest` settings are defined in [`../pyproject.toml`](../pyproject.toml).
- Pass `-p no:cacheprovider` when executing in clean-room or CI environments to avoid creating `.pytest_cache/`.

## Related Documentation

- [Reproducibility Guide](../provenance/REPRODUCIBILITY.md) · [Source Modules](../src/README.md) · [Scripts Index](../scripts/README.md)
