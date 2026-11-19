# Source Modules (`src/`)

Core reusable Python packages implementing the DERMA-ACT V2 active-perception operators, multi-head surrogate policy, split-conformal selector, shared domain constants, and data augmentation pipelines.

## What This Directory Contains

| Subpackage | Key Files | Public APIs & Responsibility |
| :--- | :--- | :--- |
| `src/core/` | `config.py`, `constants.py`, `dtos.py`, `exceptions.py`, `interfaces.py`, `logging.py` | `AppConfig`, `get_config`, `LesionClass`, `UrgencyLevel`, ImageNet `NORMALIZE_MEAN`/`NORMALIZE_STD`, typed DTOs & exceptions |
| `src/modules/active_perception/` | `action_space.py`, `actions.py`, `view_generator.py`, `policy.py`, `utility.py`, `evidence_state.py`, `cost_model.py`, `information_gain.py`, `next_best_observation.py`, `observation_encoder.py` | `ObservationAction` enum, `A0`–`A9` token/index converters, `ViewGenerator` (`A0`–`A9` OpenCV operators), `MultiHeadActionPolicy` (`6,787` params), `compute_utility`, `FixedCostModel` |
| `src/modules/selective_decision/` | `conformal.py`, `abstention.py`, `risk_controller.py`, `stopping_policy.py` | `ConformalPredictor` (split-conformal LAC `calibrate`, `predict_set`, `coverage`), `SelectiveDecision`, `RiskController`, `StoppingPolicy`, `AbstentionTracker` |
| `src/training/` | `augmentation.py` | `get_train_transforms`, `get_val_transforms`, `get_inference_transforms`, `get_tta_transforms`, `MixUpTransform`, `CutMixTransform` |

*(In the full local research workspace, `src/` also includes historical application scaffolding such as `api/`, `cli/`, `deployment/`, `domain/`, `evaluation/`, `framework/`, `infrastructure/`, `models/`, and `utils/`; only the audited DERMA-ACT V2 core subset above is exported into the clean public repository.)*

## What This Directory Does Not Contain

- Frozen DERMA-ACT V1/V2 benchmark hyperparameters (`lambda`, `tau`, SHA-256 digests) or recursive arithmetic fusion math — those live in [`research/derma_act/`](../research/README.md).
- Binary checkpoints (`.ckpt`, `.pt`), datasets, or experiment outputs.

## Dependencies

Requires `torch`, `numpy`, `opencv-python`, `albumentations`, `pydantic`, `PyYAML`, and `loguru` (see [`requirements.txt`](../requirements.txt)).

## Running Verification Tests

```bash
# Syntax compilation check [VERIFIED]
python -m compileall -q src

# Canonical math & policy architecture smoke tests [VERIFIED]
python -m pytest -p no:cacheprovider -v tests/test_canonical_math.py tests/test_release_smoke.py
```

## Inputs, Outputs & Related Documentation

- **Inputs:** RGB `uint8` image arrays `(H, W, 3)` for `ViewGenerator`; `(B, 28)` state tensors and `(B, 9)` action one-hot tensors for `MultiHeadActionPolicy`; `(N, 7)` calibrated softmax arrays and integer class labels for `ConformalPredictor`.
- **Outputs:** Deterministic `(224, 224, 3)` views, scalar `(dq_hat, dc_hat, i_logit)` predictions, and prediction sets `list[set[int]]`.
- **Related Docs:** [Architecture](../docs/ARCHITECTURE.md) · [Research Modules](../research/README.md) · [Tests](../tests/README.md)
