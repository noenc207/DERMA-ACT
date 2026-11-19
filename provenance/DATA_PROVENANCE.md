# Data Provenance (`DERMA-ACT V2`)

## 1. Primary Benchmark Dataset: ISIC 2019 Training Set
- **Authoritative Source:** International Skin Imaging Collaboration (ISIC) 2019 Challenge Training Dataset ([https://challenge.isic-archive.com/data/#2019](https://challenge.isic-archive.com/data/#2019)).
- **License & Redistribution Terms:** Creative Commons Attribution–NonCommercial 4.0 International (**`CC-BY-NC 4.0`**). Raw images (`ISIC_2019_Training_Input/*.jpg`) and contributor metadata (`ISIC_2019_Training_Metadata.csv`, `ISIC_2019_Training_GroundTruth.csv`) are **not redistributed in Git commits**. Users must acquire them directly from the official portal and comply with `CC-BY-NC 4.0` attribution for `BCN_20000`, `HAM10000`, and `MSK`.
- **Class Labels (`7`-Class Benchmark Formulation):** `MEL` (`0`, Melanoma, `4,522`), `NV` (`1`, Melanocytic Nevi, `12,875`), `BCC` (`2`, Basal Cell Carcinoma, `3,323`), `AKIEC` (`3`, merged Actinic Keratosis `AK` [`867`] + Squamous Cell Carcinoma `SCC` [`628`] = `1,495`), `BKL` (`4`, Benign Keratosis, `2,624`), `DF` (`5`, Dermatofibroma, `239`), `VASC` (`6`, Vascular Lesions, `253`). ISIC 2019's `UNK` column has `0` positive entries in the training ground-truth table.

## 2. Frozen Partition Counts (`data/isic2019/splits/split_metadata.json`, `seed = 42`, `split_hash = df52266ea5c2a4b1`)
| Split | Images (`N`) | Operational Groups (`K`) | Target Ratio |
| :--- | ---: | ---: | ---: |
| Train | 16,702 | 9,198 | 0.6604 |
| Validation (`Val`) | 3,542 | 1,972 | 0.1415 |
| Calibration (`Cal` = `D_fit` 698 + `D_conf` 703) | 1,401 | 789 | 0.0566 |
| Held-Out Test (`Test`) | 3,686 | 1,972 | 0.1415 |
| **Total ISIC 2019** | **25,331** | **13,931** | **1.0000** |

## 3. Operational Grouping Rule & Patient-Level Limitation
- **Grouping Rule:** `ISIC_2019_Training_Metadata.csv` supplies a non-null `lesion_id` for `23,247 / 25,331` images (`11,847` unique observed `lesion_id` clusters). The remaining `2,084` images lack `lesion_id` and are grouped operationally as `image_id` singletons, yielding `13,931` operational groups (`K = 1,972` on held-out Test: `1,695` observed `lesion_id` clusters + `277` missing-ID singletons).
- **Patient-Level Independence Limitation:** Patient identifiers are absent from ISIC 2019 metadata. Therefore, **patient-level independence across distinct lesions is NOT established**, and cluster-stratified bootstrap confidence intervals account only for known `lesion_id` plus singleton operational grouping.

## 4. Excluded Non-ISIC & Placeholder Datasets
- **`data/test_dataset*`:** Synthetic `100x100` (`std = 0.0`) pipeline smoke placeholders (`4`, `6`, `60` images) — not scientific data and excluded from V2 staging.
- **`data/external/pad_ufes_20*`:** Prior smartphone execution (`N = 100`, `32.00%` accuracy / domain collapse) invalidated by fallback/random weights and incorrect resizing — excluded; PAD-UFES-20 must **not** be cited as successful external validation.
- **HIBA (`N = 15`):** Runtime crash-free latency check (`0.868s` mean latency) lacking 7-class diagnostic accuracy validation — excluded as efficacy evidence.
- **`test_images/1_Melanoma_UngThuHacTo.jpg`:** Standalone natural lesion image with unverified copyright/consent — excluded from V2 staging.
