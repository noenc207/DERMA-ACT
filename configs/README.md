# Configuration Presets (`configs/`)

Hydra-compatible YAML configurations governing dataset schemas, preprocessing, model architecture, optimizer, scheduler, trainer settings, and experiment presets.

## Directory Layout

| Path | Key Files | Purpose |
| :--- | :--- | :--- |
| `configs/train.yaml` | `train.yaml` | Root Hydra training configuration (`seed: 42`, defaults for model, dataset, trainer, optimizer, scheduler, loss) |
| `configs/dataset/` | `isic2019.yaml`, `ham10000.yaml`, `default.yaml` | Dataset paths, `image_size: 224`, batch size (`32`), weighted sampler, split ratios |
| `configs/model/` | `efficientnet_b4.yaml` | Backbone specification (`efficientnet_b4`, `num_classes: 7`, `drop_rate: 0.3`) |
| `configs/augmentation/` | `standard.yaml` | Training augmentation configuration |
| `configs/preprocessing/` | `default.yaml` | Normalization and resizing parameters |
| `configs/optimizer/` | `adamw.yaml` | AdamW defaults |
| `configs/scheduler/` | `cosine_annealing.yaml` | Cosine annealing scheduler defaults |
| `configs/trainer/` | `default.yaml` | Mixed precision (`16-mixed`), gradient clipping (`1.0`), deterministic flag |
| `configs/experiment/` | `derma_act_baseline.yaml`, `default.yaml` | Experiment preset overrides |

## Critical Provenance Notice: Static YAML Defaults vs. Verified Executed Protocol

As audited in `supplementary/tables/tab_s1_classifier_protocol.tex` and `research/derma_act/config.py`, several static YAML fields in `configs/train.yaml` and `configs/dataset/isic2019.yaml` differ from the **verified executed values** used to produce the frozen EfficientNet-B4 checkpoint and ISIC 2019 splits:

| Parameter | Static YAML Value | Verified Executed Value | Authoritative Source |
| :--- | :--- | :--- | :--- |
| Learning rate (`AdamW`) | `1.0e-3` (`train.yaml`) | **`1.0e-4`** | Executed training code & checkpoint metadata (`tab_s1`) |
| Loss function | `focal` (`train.yaml`) | **`CrossEntropyLoss(label_smoothing=0.1)`** | Executed training code & checkpoint metadata (`tab_s1`) |
| Warmup / EMA | `warmup_epochs: 5`, `ema.enabled: true` | **Absent / not executed** | Checkpoint metadata (`tab_s1`) |
| Partition ratios | `0.70 / 0.10 / 0.05 / 0.15` (`isic2019.yaml`) | **`0.6604 / 0.1415 / 0.0566 / 0.1415`** (`16,702 / 3,542 / 1,401 / 3,686`) | `data/isic2019/splits/split_metadata.json` (`split_hash=df52266ea5c2a4b1`) |
| Policy hyperparameters | — | **`LAMBDA = (5.0, 1.0, 0.5, 0.25)`**, **`TAU = -3/7` (`-0.4286`)** | `research/derma_act/config.py` & `runs/derma_act_v1/policy/*.json` |

## What This Directory Does Not Contain

- Secrets, API keys, credentials, or workstation-specific absolute paths (always use `.env.example` as a template for local environment variables).

## Inspecting Configurations Programmatically

```bash
# Parse and verify all YAML configuration files cleanly [VERIFIED]
python -c "import yaml, pathlib; fs=sorted(pathlib.Path('configs').rglob('*.yaml')); [yaml.safe_load(f.read_text(encoding='utf-8')) for f in fs]; print(f'Validated {len(fs)} YAML files')"
```

## Related Documentation

- [Notebook 00 (Config & Quickstart)](../notebooks/00_Project_Overview_and_Quickstart.ipynb) · [Model Provenance](../provenance/MODEL_PROVENANCE.md) · [Reproducibility](../provenance/REPRODUCIBILITY.md)
