# License and Governance Status (`DERMA-ACT V2`)

> **No repository-wide open-source license has been established.**

This public repository contains the **DERMA-ACT V2** research codebase, experimental configurations, pure aggregate benchmark summaries, and regenerable benchmark LaTeX tables (`supplementary/tables/`). Unpublished manuscript drafts (`PAPER_DRAFT_Q2_V3_2` frozen archive and `PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION`) and journal submission documents are excluded and maintained privately outside `noenc207/DERMA-ACT`. Because legacy packaging previously referenced an unverified MIT license without a `LICENSE` file on disk—and because derivative research artifacts inherit third-party non-commercial restrictions—each asset layer is separated explicitly below.

## Layer-by-Layer Legal Separation

| Asset Layer | Scope | Verified License / Rights Status | Git Synchronization Policy |
| :--- | :--- | :--- | :--- |
| **1. Original Author Source Code** | `src/modules/active_perception/`, `research/derma_act/core/`, `scripts/`, `tests/`, `configs/` | **No repository-wide open-source license has been established** (`LICENSE` file not selected yet) | Included in this repository without an open-source license grant (all rights reserved until the author selects and commits a `LICENSE` file) |
| **2. Third-Party Code Dependencies** | PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, OpenCV, Albumentations, Pillow, Pydantic, PyYAML, Pytest, Loguru, tqdm | Governed by each upstream package's respective open-source license (BSD/MIT/Apache-2.0); **not relicensed** | Referenced via `requirements.txt` and `pyproject.toml` only; virtualenvs excluded |
| **3. Upstream Dataset Assets (`ISIC 2019`)** | `ISIC_2019_Training_Input` images & ground-truth/metadata CSVs | **`CC-BY-NC 4.0`** (Creative Commons Attribution–NonCommercial 4.0 International); includes `BCN_20000`, `HAM10000`, `MSK` | **Excluded from Git**; acquire directly from [ISIC 2019 Challenge Portal](https://challenge.isic-archive.com/data/#2019) |
| **4. Derivative Model Weights & Caches** | `.ckpt` EfficientNet-B4 weights, `policy.pt`, per-image `.npz` logit caches | Derived from `CC-BY-NC 4.0` ISIC 2019 training data; cannot be granted commercial rights (`MIT`/`CC-BY`) | **Excluded from Git**; expected SHA-256 digests documented in [provenance/MODEL_PROVENANCE.md](provenance/MODEL_PROVENANCE.md) |
| **5. Standalone Non-Canonical Images** | `test_images/1_Melanoma_UngThuHacTo.jpg`, `invalid_cohort_picsum/`, `data/external/pad_ufes_20*`, `data/test_dataset*/` | Unverified copyright, source consent, or external non-commercial terms | **Excluded** from this public repository (`.gitignore`); never redistributed in `noenc207/DERMA-ACT` |
| **6. Manuscript & Journal Policy** | Regenerable benchmark LaTeX tables (`supplementary/tables/*.tex`); unpublished manuscript sources (`main.tex`, `supplementary_materials.tex`, `.docx`, `.pdf`) | Authorship (`Duc-Son Le`) and target journal terms (*Biomedical Signal Processing and Control*, Elsevier) upon submission | Regenerable benchmark LaTeX tables (`supplementary/tables/`) included; unpublished manuscript drafts and submission documents excluded |

## Mandatory Upstream ISIC 2019 Attribution

When reproducing data-dependent (`REQUIRES_DATA`) evaluations using ISIC 2019 data, users must preserve the upstream `CC-BY-NC 4.0` notice and cite:
- **BCN_20000 Dataset:** © Department of Dermatology, Hospital Clínic de Barcelona (`Hernández-Pérez et al., Sci. Data 2024`)
- **HAM10000 Dataset:** © ViDIR Group, Department of Dermatology, Medical University of Vienna (`https://doi.org/10.1038/sdata.2018.161`)
- **MSK Dataset:** © Anonymous (`https://arxiv.org/abs/1710.05006`; `https://arxiv.org/abs/1902.03368`)

## Pending Owner Governance Actions

1. Select and commit an official software `LICENSE` for original code modules (and update `pyproject.toml`, `CITATION.cff`, and `README.md` accordingly).
2. Decide non-commercial archival hosting (`CC-BY-NC 4.0` or controlled access, e.g., Zenodo DOI) for excluded model checkpoints and row-level `.npz` bundles.
3. Keep unpublished manuscript drafts, journal submission documents, restricted datasets (`ISIC 2019`, `PAD-UFES-20`, `HIBA`), model checkpoints, and local credentials outside `noenc207/DERMA-ACT` (verified excluded at release time).
