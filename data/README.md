# Data Layout, Governance & Acquisition (`data/`)

Documents the tracked non-sensitive benchmark summaries inside `data/` and the required local placement for restricted upstream ISIC 2019 files acquired by the user.

## What This Directory Contains (Tracked in Git)

| Path | Description | Contains Patient/Image Pixels? |
| :--- | :--- | :---: |
| `data/isic2019/splits/split_metadata.json` | Frozen group-disjoint partition census (`25,331` images; `13,931` operational groups; `seed = 42`; `split_hash = df52266ea5c2a4b1`) | No (counts & config only) |
| `data/aggregate/FIGURE_4_V1_1_DATA.csv` | Aggregate multi-view risk-cost benchmark table across budgets `B = 0..5` with cluster-bootstrap CIs and Holm–Bonferroni $p$-values | No (aggregate summary only) |

## What This Directory Does Not Contain (Excluded from Public Distribution)

- **ISIC 2019 Raw Imagery & CSVs (`CC-BY-NC 4.0`):** Neither `ISIC_2019_Training_Input/*.jpg` nor `ISIC_2019_Training_Metadata.csv` / `ISIC_2019_Training_GroundTruth.csv` are bundled or redistributed.
- **External & Placeholder Datasets:** Synthetic smoke placeholders (`test_dataset*`), unverified standalone images, invalidated PAD-UFES-20 runs, and non-diagnostic HIBA latency logs are strictly excluded.

## Expected Local Directory Layout (`REQUIRES_DATA`)

Acquire the official ISIC 2019 Training dataset directly from [https://challenge.isic-archive.com/data/#2019](https://challenge.isic-archive.com/data/#2019) under `CC-BY-NC 4.0` terms and place files locally (ignored by `.gitignore`):

```text
data/
├── isic2019/
│   ├── ISIC_2019_Training_Input/      # 25,331 JPEG dermoscopic images (gitignored)
│   ├── raw/
│   │   ├── ISIC_2019_Training_GroundTruth.csv
│   │   └── ISIC_2019_Training_Metadata.csv
│   └── splits/
│       ├── split_metadata.json        # Tracked
│       ├── train_indices.csv          # Local (16,702 image IDs)
│       ├── val_indices.csv            # Local (3,542 image IDs)
│       ├── cal_indices.csv            # Local (1,401 image IDs)
│       └── test_indices.csv           # Local (3,686 image IDs)
└── aggregate/
    └── FIGURE_4_V1_1_DATA.csv         # Tracked
```

## Operational Grouping & Attribution

- **Operational Groups (`13,931` total):** `23,247 / 25,331` images carry an observed `lesion_id` (`11,847` unique clusters); `2,084` images lack `lesion_id` and are treated operationally as `image_id` singletons. Patient-level identifiers are absent in ISIC 2019.
- **Mandatory Attributions:** Preserve `CC-BY-NC 4.0` notices for `BCN_20000`, `HAM10000`, and `MSK` (exact wording in [`../docs/DATA_ACQUISITION_AND_ATTRIBUTION.md`](../docs/DATA_ACQUISITION_AND_ATTRIBUTION.md)).

## Verification

Inspect schema and partition counts interactively using [`../notebooks/01_ISIC2019_Data_and_Split_Audit.ipynb`](../notebooks/01_ISIC2019_Data_and_Split_Audit.ipynb).

## Related Documentation

- [Data Acquisition & Attribution](../docs/DATA_ACQUISITION_AND_ATTRIBUTION.md) · [Data Provenance](../provenance/DATA_PROVENANCE.md) · [License Status](../LICENSE_STATUS.md)
