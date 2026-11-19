# Provenance Documentation (`provenance/`)

Authoritative traceability records linking the DERMA-ACT V2 codebase, upstream ISIC 2019 dataset, frozen model artifacts, aggregate experimental ledgers, and verified execution commands.

## Contents

| Document | Scope |
| :--- | :--- |
| [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) | ISIC 2019 7-class mapping (`AK`+`SCC` -> `AKIEC`), frozen partition counts (`split_hash = df52266ea5c2a4b1`), operational grouping rule (`11,847` observed `lesion_id` + `2,084` missing-ID singletons), and excluded non-canonical datasets |
| [`MODEL_PROVENANCE.md`](MODEL_PROVENANCE.md) | Frozen EfficientNet-B4 checkpoint digest (`0d14579d...`), historical training seed caveat, temperature calibration ($T_{\mathrm{full}} = 0.8518$), `MultiHeadActionPolicy` (`6,787` params, `policy.pt` digest `32a098c8...`), and split/Mondrian conformal summaries |
| [`EXPERIMENT_PROVENANCE.md`](EXPERIMENT_PROVENANCE.md) | Mapping of canonical aggregate CSV/JSON ledgers under `research/derma_act/` to manuscript claims and explicit exclusion of superseded runs / `mlflow.db` |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Verified Python `3.11.9` / PyTorch `2.5.1+cu121` package matrix, deterministic seeds (`42`, `20261112`), `VERIFIED` standalone commands, and `REQUIRES_DATA` boundaries |

## What This Directory Does Not Contain

- Raw data, binary weights, or historical security audit logs (stored privately outside public release).

## Related Documentation

- [Documentation Index](../docs/README.md) · [Artifacts Registry](../artifacts/README.md) · [License Status](../LICENSE_STATUS.md)
