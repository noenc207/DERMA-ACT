# Data Acquisition & Attribution (`ISIC 2019`)

Raw dermoscopic images, per-image metadata CSVs, trained model checkpoints (`.ckpt`, `.pt`), and row-level cached tensor outputs (`.npz`, per-image `.csv` traces) are **excluded from Git commits** to comply with upstream licensing and data governance boundaries.

---

## 1. Upstream Dataset Source & License

- **Dataset:** **ISIC 2019 Challenge Training Set** (`25,331` JPEG dermoscopic images across `8` active diagnostic columns + zero-count `UNK`; `7`-class benchmark formulation merging `AK` [`867`] + `SCC` [`628`] into `AKIEC` [`1,495`])
- **Official Challenge Portal:** [https://challenge.isic-archive.com/data/#2019](https://challenge.isic-archive.com/data/#2019)
- **Official License:** **`CC-BY-NC 4.0`** (Creative Commons Attribution–NonCommercial 4.0 International)
- **Terms of Use:** [https://challenge.isic-archive.com/terms-of-use/](https://challenge.isic-archive.com/terms-of-use/)

> **Non-Commercial & Redistribution Notice:** Any user downloading ISIC 2019 imagery directly from the Challenge portal, or training/evaluating derivative representations and checkpoints on it, remains legally bound by the **`CC-BY-NC 4.0`** terms and contributor licenses. See [../LICENSE_STATUS.md](../LICENSE_STATUS.md) regarding pending repository-wide code licensing.

---

## 2. Mandatory Upstream Copyright Attributions (`ISIC 2019: Training`)

As mandated verbatim by [https://challenge.isic-archive.com/data/#2019](https://challenge.isic-archive.com/data/#2019) to satisfy `CC-BY-NC` attribution:

- **BCN_20000 Dataset:** © Department of Dermatology, Hospital Clínic de Barcelona
- **HAM10000 Dataset:** © by ViDIR Group, Department of Dermatology, Medical University of Vienna (`https://doi.org/10.1038/sdata.2018.161`)
- **MSK Dataset:** © Anonymous (`https://arxiv.org/abs/1710.05006`; `https://arxiv.org/abs/1902.03368`)

And mandatory full citations:
1. Tschandl P., Rosendahl C., & Kittler H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data **5**, 180161 (`doi:10.1038/sdata.2018.161`, 2018).
2. Codella N. C. F., Gutman D., Celebi M. E., Helba B., Marchetti M. A., Dusza S. W., Kalloo A., Liopyris K., Mishra N., Kittler H., & Halpern A. *Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC).* `arXiv:1710.05006` (2017). *(With ISIC 2019 Task overview: Combalia et al., `arXiv:1902.03368`, 2019).*
3. Hernández-Pérez C., Combalia M., Podlipnik S., Codella N. C., Rotemberg V., Halpern A. C., Reiter O., Carrera C., Barreiro A., Helba B., Puig S., Vilaplana V., & Malvehy J. *BCN20000: Dermoscopic lesions in the wild.* Scientific Data **11**(1), 641 (2024).

---

## 3. Expected Artifact Placement for Data-Dependent Evaluation (`REQUIRES_DATA`)

When running data-dependent scripts (`--root <DATA_ROOT>`), acquire ISIC 2019 data and authorized non-commercial frozen artifacts into local workspace paths excluded by `.gitignore`:

- `data/isic2019/raw/ISIC_2019_Training_Metadata.csv` and raw `.jpg` images (`ISIC_0012970.jpg` for Figure 2, etc.)
- `research/baseline_v2/runs/clean_run_001/predictions/{calibration_canonical.npz,test_canonical.npz}`
- `research/baseline_v2/runs/clean_run_001/conformal/test_prediction_sets.npz`
- `research/derma_act/runs/derma_act_v0/predictions/per_action_test.npz`
- `research/derma_act/evidence/v1/{per_action_dev.npz,test_policy_trace.csv}`
- `research/derma_act/runs/derma_act_v1/policy/policy.pt` (`sha256: 32a098c8e2aa958959122cb0292eff0cc8932ea4dac4c2599f3dd37ed2587c95`)
- `checkpoints/baseline_v2/clean_run_001/epoch=09-val_auroc=0.0000.ckpt` (`sha256: 0d14579d668d02781874cf91ad493fac4bf33ed53e3be928e3a7f9bb856ebbbb`)

> **Public Data Repository Status:** **PUBLIC ARCHIVE HOSTING PENDING / NO DOI MINTED.** Availability of packaged heavy `.npz` / `.ckpt` bundles remains pending non-commercial archive setup (see [../LICENSE_STATUS.md](../LICENSE_STATUS.md)).
