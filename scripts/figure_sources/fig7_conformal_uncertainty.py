"""Deterministic Figure 7 V1.1 generator for DERMA-ACT V2 (P7.1 Corrected Quantile).

Uses the exact finite-sample order statistic q_star = sorted_scores[k - 1]
with k = ceil((N_conf + 1) * (1 - alpha)) = 634 (0-based index 633) on
D_conf (N_conf = 703, K_conf = 396), correcting the +1 rank shift induced by
np.quantile(..., method="higher") in the superseded Pilot.

Zero model training, zero fine-tuning, and zero classifier inference are performed.
"""

from __future__ import annotations

import csv

import argparse as _fg_argparse
from pathlib import Path as _FgPath
_fg_ap = _fg_argparse.ArgumentParser(description='Figure 7: Split-Conformal Prediction Sets (Tier B canonical logits & ISIC metadata required)')
_fg_ap.add_argument('--root', type=_FgPath, default=_FgPath(__file__).resolve().parents[2], help='Data/repository root')
_fg_ap.add_argument('--out-dir', type=_FgPath, default=_FgPath(__file__).resolve().parents[2] / 'outputs/figures', help='Output figure directory')
_fg_args, _ = _fg_ap.parse_known_args()
_FG_ROOT = _fg_args.root.resolve()
_FG_OUT = _fg_args.out_dir.resolve()
_FG_OUT.mkdir(parents=True, exist_ok=True)

import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.special import softmax

ROOT = _FG_ROOT
RUN = ROOT / "research/baseline_v2/runs/clean_run_001"
OUT_DIR = _FG_OUT

ALPHA = 0.10
SPLIT_SEED = 20261112
BOOT_SEED = 20261112
N_BOOT = 2000
N_SPLITS_SENS = 1000

CLASS_CODES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_NAMES = {
    "mel": "Melanoma (MEL)",
    "nv": "Melanocytic Nevus (NV)",
    "bcc": "Basal Cell Carcinoma (BCC)",
    "akiec": "Actinic Keratosis / IEC (AKIEC)",
    "bkl": "Benign Keratosis (BKL)",
    "df": "Dermatofibroma (DF)",
    "vasc": "Vascular Lesion (VASC)",
}
SHORT_LABELS = {
    "akiec": "AKIEC\n(Premalignant)",
    "df": "DF",
    "bkl": "BKL",
    "mel": "MEL",
    "bcc": "BCC",
    "nv": "NV",
    "vasc": "VASC\n(Sparse n=19)",
}


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Fit scalar temperature T > 0 by NLL minimization (BFGS on log T)."""

    def nll(log_t: np.ndarray) -> float:
        t = float(np.exp(log_t[0]))
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-log_p[np.arange(len(y)), y].mean())

    res = minimize(nll, [0.0], method="BFGS")
    return float(np.exp(res.x[0]))


def build_operational_groups(image_ids: np.ndarray, meta_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Map images to known lesion_id groups or singleton 'IMG::' tokens."""
    groups = []
    is_missing = []
    for img_id in image_ids:
        lid = meta_df.lesion_id.get(img_id)
        if isinstance(lid, str) and len(lid.strip()) > 0:
            groups.append(lid)
            is_missing.append(False)
        else:
            groups.append(f"IMG::{img_id}")
            is_missing.append(True)
    return np.array(groups), np.array(is_missing, dtype=bool)


def group_stratified_split(
    cal_grp: np.ndarray, yc: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic majority-class stratified 50/50 split over calibration groups."""
    ugc = np.unique(cal_grp)
    g_lab = np.array([np.bincount(yc[cal_grp == g], minlength=7).argmax() for g in ugc])
    rng = np.random.default_rng(seed)
    fit_groups = []
    for c in range(7):
        gs = ugc[g_lab == c]
        rng.shuffle(gs)
        fit_groups.extend(gs[: len(gs) // 2])
    fit_mask = np.isin(cal_grp, np.array(fit_groups))
    return np.where(fit_mask)[0], np.where(~fit_mask)[0]


def image_stratified_split(yc: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Legacy P5.2 image-level stratified 50/50 split."""
    rng = np.random.default_rng(seed)
    fit_idx, conf_idx = [], []
    for c in range(7):
        idx = np.where(yc == c)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        fit_idx.extend(idx[:half])
        conf_idx.extend(idx[half:])
    return np.sort(np.array(fit_idx)), np.sort(np.array(conf_idx))


def conformal_threshold(
    scores: np.ndarray, alpha: float, quantile_mode: str
) -> tuple[float, int, float]:
    """Compute finite-sample conformal quantile."""
    n = len(scores)
    k = math.ceil((n + 1) * (1.0 - alpha))
    q_level = min(k / n, 1.0)
    if quantile_mode == "sort_k":
        return float(np.sort(scores)[k - 1]), k, q_level
    return float(np.quantile(scores, q_level, method=quantile_mode)), k, q_level


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% binomial CI (reference for sparse categories)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def evaluate_protocol(
    name: str,
    description: str,
    T_fit: float,
    q_hat: float,
    rank_used: int,
    Lt: np.ndarray,
    yt: np.ndarray,
    t_members: list[np.ndarray],
    n_fit: int,
    k_fit: int,
    n_conf: int,
    k_conf: int,
    n_boot: int = N_BOOT,
    boot_seed: int = BOOT_SEED,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a conformal threshold on frozen held-out test logits with cluster bootstrap."""
    p_te = softmax(Lt / T_fit, axis=1)
    pred_sets = (1.0 - p_te) <= q_hat
    assert np.array_equal(pred_sets, p_te >= (1.0 - q_hat)), "Predicate boundary mismatch!"
    cov_vec = pred_sets[np.arange(len(yt)), yt]
    sz_vec = pred_sets.sum(axis=1)

    n_clusters = len(t_members)
    rng = np.random.default_rng(boot_seed)
    boot_cov = np.empty(n_boot)
    boot_sz = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([t_members[g] for g in pick])
        boot_cov[b] = cov_vec[idx].mean()
        boot_sz[b] = sz_vec[idx].mean()

    row = {
        "protocol_id": name,
        "description": description,
        "n_fit": n_fit,
        "k_fit": k_fit,
        "n_conf": n_conf,
        "k_conf": k_conf,
        "rank_1based": rank_used,
        "T_fit": round(T_fit, 8),
        "q_hat": round(q_hat, 8),
        "tau_inclusion": round(1.0 - q_hat, 8),
        "covered_images": int(cov_vec.sum()),
        "n_test_images": int(len(yt)),
        "marginal_coverage": round(float(cov_vec.mean()), 6),
        "ci95_low": round(float(np.percentile(boot_cov, 2.5)), 6),
        "ci95_high": round(float(np.percentile(boot_cov, 97.5)), 6),
        "mean_set_size": round(float(sz_vec.mean()), 6),
        "mean_set_size_sd": round(float(sz_vec.std(ddof=1)), 6),
        "set_size_ci95_low": round(float(np.percentile(boot_sz, 2.5)), 6),
        "set_size_ci95_high": round(float(np.percentile(boot_sz, 97.5)), 6),
        "singleton_count": int((sz_vec == 1).sum()),
        "singleton_rate": round(float((sz_vec == 1).mean()), 6),
        "empty_count": int((sz_vec == 0).sum()),
    }
    for s in range(8):
        row[f"size_{s}_count"] = int((sz_vec == s).sum())
        row[f"size_{s}_pct"] = round(float((sz_vec == s).mean() * 100.0), 4)
    return row, pred_sets, cov_vec, sz_vec


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ca = np.load(RUN / "predictions/calibration_canonical.npz")
    te = np.load(RUN / "predictions/test_canonical.npz")
    stored_sets = np.load(RUN / "conformal/test_prediction_sets.npz")
    meta = pd.read_csv(ROOT / "data/isic2019/raw/ISIC_2019_Training_Metadata.csv").set_index("image")

    Lc = ca["logits"].astype(np.float64)
    yc = ca["true_label"].astype(np.int64)
    Lt = te["logits"].astype(np.float64)
    yt = te["true_label"].astype(np.int64)

    cal_grp, _ = build_operational_groups(ca["image_id"], meta)
    te_grp, te_miss = build_operational_groups(te["image_id"], meta)

    utg, tginv = np.unique(te_grp, return_inverse=True)
    t_members = [np.where(tginv == g)[0] for g in range(len(utg))]

    # 1. Corrected Primary Group-Disjoint Split (P7.1: exact k=634 -> sorted_scores[633])
    fi_grp, ci_grp = group_stratified_split(cal_grp, yc, SPLIT_SEED)
    T_grp = fit_temperature(Lc[fi_grp], yc[fi_grp])
    sc_grp = 1.0 - softmax(Lc[ci_grp] / T_grp, axis=1)[np.arange(len(ci_grp)), yc[ci_grp]]

    q_star, k_star, _ = conformal_threshold(sc_grp, ALPHA, "sort_k")
    q_pilot_higher, _, _ = conformal_threshold(sc_grp, ALPHA, "higher")

    k_fit_grp = int(np.unique(cal_grp[fi_grp]).size)
    k_conf_grp = int(np.unique(cal_grp[ci_grp]).size)

    row_primary, sets_primary, cov_primary, sz_primary = evaluate_protocol(
        "group_disjoint_p71_corrected_k634",
        "Primary P7.1 corrected group-disjoint split (seed=20261112, LAC, exact k=634 sorted_scores[633])",
        T_grp,
        q_star,
        k_star,
        Lt,
        yt,
        t_members,
        n_fit=len(fi_grp),
        k_fit=k_fit_grp,
        n_conf=len(ci_grp),
        k_conf=k_conf_grp,
    )

    row_superseded_pilot, _, _, _ = evaluate_protocol(
        "group_disjoint_p53_pilot_higher_k635",
        "Superseded Figure 7 Pilot (np.quantile method='higher', selected 635th score index 634)",
        T_grp,
        q_pilot_higher,
        k_star + 1,
        Lt,
        yt,
        t_members,
        n_fit=len(fi_grp),
        k_fit=k_fit_grp,
        n_conf=len(ci_grp),
        k_conf=k_conf_grp,
    )

    # Exploratory Group-Max conformal rule on D_conf groups (K_conf=396, k=358)
    conf_grp_ids = cal_grp[ci_grp]
    ucg, cginv = np.unique(conf_grp_ids, return_inverse=True)
    grp_max_scores = np.array([sc_grp[cginv == g].max() for g in range(len(ucg))])
    q_grp_max_sort, k_grp_max, _ = conformal_threshold(grp_max_scores, ALPHA, "sort_k")

    row_grpmax_sort, sets_grpmax, cov_grpmax, _ = evaluate_protocol(
        "group_max_k396_sort_k358",
        "Exploratory cluster-max score aggregation on D_conf (K_conf=396 groups, exact k=358)",
        T_grp,
        q_grp_max_sort,
        k_grp_max,
        Lt,
        yt,
        t_members,
        n_fit=len(fi_grp),
        k_fit=k_fit_grp,
        n_conf=len(ci_grp),
        k_conf=k_conf_grp,
    )

    # Legacy P5.2 image-stratified & Legacy full-CAL reused
    fi_img, ci_img = image_stratified_split(yc, SPLIT_SEED)
    T_img = fit_temperature(Lc[fi_img], yc[fi_img])
    sc_img = 1.0 - softmax(Lc[ci_img] / T_img, axis=1)[np.arange(len(ci_img)), yc[ci_img]]
    q_img_lin, k_img_ord, _ = conformal_threshold(sc_img, ALPHA, "linear")
    row_img_split, _, _, _ = evaluate_protocol(
        "image_disjoint_p52_linear",
        "Legacy P5.2 image-stratified split (699/702, shares 184 lesion_ids, linear quantile)",
        T_img,
        q_img_lin,
        k_img_ord,
        Lt,
        yt,
        t_members,
        n_fit=len(fi_img),
        k_fit=int(np.unique(cal_grp[fi_img]).size),
        n_conf=len(ci_img),
        k_conf=int(np.unique(cal_grp[ci_img]).size),
    )

    T_full = fit_temperature(Lc, yc)
    sc_full = 1.0 - softmax(Lc / T_full, axis=1)[np.arange(len(yc)), yc]
    q_full_lin, k_full_ord, _ = conformal_threshold(sc_full, ALPHA, "linear")
    row_reused, sets_reused, _, _ = evaluate_protocol(
        "reused_full_cal_legacy_linear",
        "Legacy clean_run_001 baseline (reuses N=1,401 for both T and q_hat, linear quantile)",
        T_full,
        q_full_lin,
        k_full_ord,
        Lt,
        yt,
        t_members,
        n_fit=len(yc),
        k_fit=int(np.unique(cal_grp).size),
        n_conf=len(yc),
        k_conf=int(np.unique(cal_grp).size),
    )
    assert np.array_equal(sets_reused, stored_sets["prediction_sets"]), "Stored legacy sets mismatch!"

    # 1,000 random group-disjoint split sensitivity using exact sort_k (seeds 0..999)
    split_covs = np.empty(N_SPLITS_SENS)
    split_sizes = np.empty(N_SPLITS_SENS)
    for sd in range(N_SPLITS_SENS):
        f_s, c_s = group_stratified_split(cal_grp, yc, sd)
        t_s = fit_temperature(Lc[f_s], yc[f_s])
        sc_s = 1.0 - softmax(Lc[c_s] / t_s, axis=1)[np.arange(len(c_s)), yc[c_s]]
        q_s, _, _ = conformal_threshold(sc_s, ALPHA, "sort_k")
        ps_s = (1.0 - softmax(Lt / t_s, axis=1)) <= q_s
        split_covs[sd] = ps_s[np.arange(len(yt)), yt].mean()
        split_sizes[sd] = ps_s.sum(axis=1).mean()

    panel_b_rows = [row_primary, row_superseded_pilot, row_grpmax_sort, row_img_split, row_reused]
    # csv export skipped in figure dir

    # 2. Panel C: Class-Conditional Empirical Coverage (q_star)
    rng_b = np.random.default_rng(BOOT_SEED)
    n_clusters = len(utg)
    boot_cls_cov = np.empty((N_BOOT, 7))
    boot_cls_sz = np.empty((N_BOOT, 7))
    boot_stratum_cov = np.empty((N_BOOT, 6))

    all_grp_cov = np.array([cov_primary[m].all() for m in t_members])
    any_grp_cov = np.array([cov_primary[m].any() for m in t_members])
    grp_sizes = np.array([len(m) for m in t_members])
    grp_is_missing = np.array([te_miss[m[0]] for m in t_members])

    is_sing_img = np.isin(te_grp, utg[grp_sizes == 1])
    is_mult_img = ~is_sing_img

    for b in range(N_BOOT):
        pick = rng_b.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([t_members[g] for g in pick])
        cv = cov_primary[idx]
        sv = sz_primary[idx]
        yy = yt[idx]
        for c in range(7):
            mc = yy == c
            boot_cls_cov[b, c] = cv[mc].mean() if mc.sum() > 0 else np.nan
            boot_cls_sz[b, c] = sv[mc].mean() if mc.sum() > 0 else np.nan

        boot_stratum_cov[b, 0] = cv.mean()
        boot_stratum_cov[b, 1] = cv[is_sing_img[idx]].mean()
        boot_stratum_cov[b, 2] = cv[is_mult_img[idx]].mean()
        boot_stratum_cov[b, 3] = all_grp_cov[pick].mean()
        pick_mult = pick[grp_sizes[pick] >= 2]
        boot_stratum_cov[b, 4] = all_grp_cov[pick_mult].mean() if len(pick_mult) > 0 else np.nan
        boot_stratum_cov[b, 5] = any_grp_cov[pick].mean()

    panel_c_rows = []
    for c, code in enumerate(CLASS_CODES):
        mc = yt == c
        n_c = int(mc.sum())
        cov_c = int(cov_primary[mc].sum())
        emp_c = cov_c / n_c
        ci_l, ci_h = np.nanpercentile(boot_cls_cov[:, c], [2.5, 97.5])
        sz_l, sz_h = np.nanpercentile(boot_cls_sz[:, c], [2.5, 97.5])
        w_l, w_h = wilson_ci(cov_c, n_c)
        n_cls_clusters = int(np.unique(te_grp[mc]).size)
        panel_c_rows.append(
            {
                "class_idx": c,
                "class_code": code.upper(),
                "class_name": CLASS_NAMES[code],
                "support_n": n_c,
                "support_pct": round(100.0 * n_c / len(yt), 4),
                "n_lesion_groups": n_cls_clusters,
                "covered_n": cov_c,
                "empirical_coverage": round(emp_c, 6),
                "cluster_ci95_low": round(float(ci_l), 6),
                "cluster_ci95_high": round(float(ci_h), 6),
                "wilson_ci95_low": round(w_l, 6),
                "wilson_ci95_high": round(w_h, 6),
                "nominal_target": 0.90,
                "coverage_gap_pp": round((emp_c - 0.90) * 100.0, 4),
                "mean_set_size": round(float(sz_primary[mc].mean()), 6),
                "set_size_ci95_low": round(float(sz_l), 6),
                "set_size_ci95_high": round(float(sz_h), 6),
                "singleton_n": int((sz_primary[mc] == 1).sum()),
                "singleton_rate": round(float((sz_primary[mc] == 1).mean()), 6),
                "empty_n": int((sz_primary[mc] == 0).sum()),
                "sparse_flag": bool(n_c < 30),
            }
        )
    panel_c_df = pd.DataFrame(panel_c_rows).sort_values("empirical_coverage", ascending=True)
    # csv export skipped in figure dir

    # 3. Panel D: Image-Weighted vs Group-Weighted Descriptive Strata (q_star)
    sing_grp_mask = grp_sizes == 1
    mult_grp_mask = grp_sizes >= 2
    is_known_img = ~te_miss
    is_miss_img = te_miss
    grp_mean_cov = np.array([cov_primary[m].mean() for m in t_members])
    all_grp_gm = np.array([cov_grpmax[m].all() for m in t_members])

    rng_d = np.random.default_rng(BOOT_SEED)
    boot_extra = np.empty((N_BOOT, 6))
    for b in range(N_BOOT):
        pick = rng_d.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([t_members[g] for g in pick])
        cv = cov_primary[idx]
        boot_extra[b, 0] = cv[is_known_img[idx]].mean()
        boot_extra[b, 1] = cv[is_miss_img[idx]].mean()
        ps_mask = pick[grp_sizes[pick] == 1]
        pm_mask = pick[grp_sizes[pick] >= 2]
        boot_extra[b, 2] = all_grp_cov[ps_mask].mean()
        boot_extra[b, 3] = any_grp_cov[pm_mask].mean()
        boot_extra[b, 4] = grp_mean_cov[pick].mean()
        boot_extra[b, 5] = all_grp_gm[pick].mean()

    def make_d_row(mid, cat, label, unit, denom, num, emp, ci_l, ci_h, note):
        return {
            "metric_id": mid,
            "estimand_type": cat,
            "metric_label": label,
            "unit": unit,
            "denominator": int(denom),
            "numerator": int(num),
            "empirical_coverage": round(float(emp), 6),
            "cluster_ci95_low": round(float(ci_l), 6),
            "cluster_ci95_high": round(float(ci_h), 6),
            "note": note,
        }

    panel_d_rows = [
        make_d_row(
            "img_all", "Image-Weighted", "All Held-Out Images", "Image (N=3,686)",
            len(yt), cov_primary.sum(), cov_primary.mean(),
            *np.percentile(boot_stratum_cov[:, 0], [2.5, 97.5]),
            "Observed empirical marginal image-level coverage (q_star = 0.904267)",
        ),
        make_d_row(
            "img_singleton_grp", "Image-Weighted", "Singleton-Group Images (k=1)", "Image (N=1,226)",
            is_sing_img.sum(), cov_primary[is_sing_img].sum(), cov_primary[is_sing_img].mean(),
            *np.percentile(boot_stratum_cov[:, 1], [2.5, 97.5]),
            "Images belonging to single-photo groups (949 known + 277 missing ID)",
        ),
        make_d_row(
            "img_multi_grp", "Image-Weighted", "Multi-Image-Group Photos (k>=2)", "Image (N=2,460)",
            is_mult_img.sum(), cov_primary[is_mult_img].sum(), cov_primary[is_mult_img].mean(),
            *np.percentile(boot_stratum_cov[:, 2], [2.5, 97.5]),
            "Correlated repeated acquisitions from 746 multi-image lesions",
        ),
        make_d_row(
            "img_known_id", "Image-Weighted", "Known Lesion-ID Images", "Image (N=3,409)",
            is_known_img.sum(), cov_primary[is_known_img].sum(), cov_primary[is_known_img].mean(),
            *np.percentile(boot_extra[:, 0], [2.5, 97.5]),
            "Subset with observed ISIC 2019 lesion_id (1,695 groups)",
        ),
        make_d_row(
            "img_missing_id", "Image-Weighted", "Missing Lesion-ID Singletons", "Image (N=277)",
            is_miss_img.sum(), cov_primary[is_miss_img].sum(), cov_primary[is_miss_img].mean(),
            *np.percentile(boot_extra[:, 1], [2.5, 97.5]),
            "Images lacking lesion_id; unverified patient/lesion linkage",
        ),
        make_d_row(
            "grp_all_cov", "Group-Weighted (Conjunction)", "All-Images-Covered (All Groups)", "Cluster (K=1,972)",
            len(utg), all_grp_cov.sum(), all_grp_cov.mean(),
            *np.percentile(boot_stratum_cov[:, 3], [2.5, 97.5]),
            "Strict conjunction across all images per group (1,770/1,972 = 89.76%)",
        ),
        make_d_row(
            "grp_all_cov_sing", "Group-Weighted (Conjunction)", "All-Covered: Singleton Groups (k=1)", "Cluster (K=1,226)",
            sing_grp_mask.sum(), all_grp_cov[sing_grp_mask].sum(), all_grp_cov[sing_grp_mask].mean(),
            *np.percentile(boot_extra[:, 2], [2.5, 97.5]),
            "Identical to image coverage on k=1 singleton units",
        ),
        make_d_row(
            "grp_all_cov_multi", "Group-Weighted (Conjunction)", "All-Covered: Multi-Image Groups (k>=2)", "Cluster (K=746)",
            mult_grp_mask.sum(), all_grp_cov[mult_grp_mask].sum(), all_grp_cov[mult_grp_mask].mean(),
            *np.nanpercentile(boot_stratum_cov[:, 4], [2.5, 97.5]),
            "Strict conjunction across all 2..24 images of multi-photo lesions (606/746)",
        ),
        make_d_row(
            "grp_any_cov", "Group-Weighted (Disjunction)", "Any-Image-Covered (All Groups)", "Cluster (K=1,972)",
            len(utg), any_grp_cov.sum(), any_grp_cov.mean(),
            *np.percentile(boot_stratum_cov[:, 5], [2.5, 97.5]),
            "At least one acquisition in the lesion group covers true label (1,870/1,972 = 94.83%)",
        ),
        make_d_row(
            "grp_any_cov_multi", "Group-Weighted (Disjunction)", "Any-Covered: Multi-Image Groups (k>=2)", "Cluster (K=746)",
            mult_grp_mask.sum(), any_grp_cov[mult_grp_mask].sum(), any_grp_cov[mult_grp_mask].mean(),
            *np.percentile(boot_extra[:, 3], [2.5, 97.5]),
            "Multi-photo lesion has >=1 covering view (706/746)",
        ),
        make_d_row(
            "grp_equal_weighted_mean", "Group-Weighted (Equal Cluster Weight)", "Equal-Cluster-Weighted Mean Coverage", "Cluster (K=1,972)",
            len(utg), round(float(grp_mean_cov.sum())), float(grp_mean_cov.mean()),
            *np.percentile(boot_extra[:, 4], [2.5, 97.5]),
            "Unweighted average of within-cluster coverage proportions across K=1,972 clusters",
        ),
        make_d_row(
            "grpmax_all_cov", "Exploratory Group-Max Rule", "Group-Max Rule All-Covered (q=0.908298)", "Cluster (K=1,972)",
            len(utg), all_grp_gm.sum(), float(all_grp_gm.mean()),
            *np.percentile(boot_extra[:, 5], [2.5, 97.5]),
            "Exploratory cluster-max calibration over K_conf=396 groups (k=358)",
        ),
    ]
    # csv export skipped in figure dir

    summary_payload = {
        "row_primary_q_star": row_primary,
        "row_superseded_pilot": row_superseded_pilot,
        "splits_1000_sort_k": {
            "coverage_mean": float(split_covs.mean()),
            "coverage_p2_5": float(np.percentile(split_covs, 2.5)),
            "coverage_p97_5": float(np.percentile(split_covs, 97.5)),
            "coverage_min": float(split_covs.min()),
            "coverage_max": float(split_covs.max()),
            "frac_below_90": float((split_covs < 0.90).mean()),
            "size_mean": float(split_sizes.mean()),
        },
    }
    print("V1_1_SUMMARY_METRICS:", json.dumps(summary_payload, indent=2))

    # =========================================================================
    # 4. RENDER PUBLICATION FIGURE 7 V1.1 (2x2 GRID, 300 DPI, PDF / SVG / PNG)
    # =========================================================================
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8.8,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.3,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.0,
            "figure.titlesize": 12.0,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 11.4))
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]

    fig.suptitle(
        "Figure 7: Group-Disjoint Split-Conformal Prediction — Finite-Sample Order Statistic (k = 634) & Validity Bounds",
        fontsize=12.8,
        fontweight="bold",
        y=0.982,
    )

    # -------------------------------------------------------------------------
    # PANEL A: Group-Disjoint Split-Conformal Protocol & Assumptions Boundary
    # -------------------------------------------------------------------------
    axA.set_xlim(0, 10)
    axA.set_ylim(0, 10)
    axA.axis("off")
    axA.set_title("A. Group-Disjoint Calibration & Finite-Sample Order Statistic (k = 634)", loc="left", pad=10)

    def draw_box(ax, x, y, w, h, title, body, fc="#F8F9FA", ec="#2B3A42", lw=1.3, title_color="#1B2631"):
        patch = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.16",
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color=title_color)
        if body:
            ax.text(x + w / 2, y + h - 0.56, body, ha="center", va="top", fontsize=7.5, color="#222222", linespacing=1.28)

    draw_box(
        axA, 0.20, 8.20, 9.60, 1.50,
        "Full Calibration Partition (D_cal): N = 1,401 Images across K = 789 Grouping Units",
        "670 observed lesion_id clusters + 119 missing-ID singleton images (0 overlap with Test or Train/Val)\n"
        "Majority-class 50/50 group split (seed=20261112) -> 0 shared images & 0 shared lesion_ids",
        fc="#EEF4FA", ec="#0072B2"
    )

    axA.annotate("", xy=(2.25, 7.62), xytext=(3.55, 8.08), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
    axA.annotate("", xy=(7.75, 7.62), xytext=(6.45, 8.08), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))

    draw_box(
        axA, 0.20, 5.55, 4.15, 1.96,
        "Temperature Fitting Subset (D_fit)",
        "N_fit = 698 images across K_fit = 393 groups\n(336 known lesions + 57 missing singletons)\n"
        "BFGS NLL minimization on frozen logits:\nFitted Scalar Temperature T_fit = 0.855690",
        fc="#F2F7FB", ec="#0072B2"
    )

    draw_box(
        axA, 5.65, 5.55, 4.15, 1.96,
        "Conformal Quantile Subset (D_conf)",
        "N_conf = 703 images across K_conf = 396 groups\n(334 known lesions + 62 missing singletons)\n"
        "k = ceil(704*0.90) = 634 -> s_(634):\nq* = 0.904267  (prob cutoff 0.095733)",
        fc="#FDF6EC", ec="#E69F00", title_color="#7A4F00"
    )

    axA.annotate("", xy=(5.50, 6.42), xytext=(4.50, 6.42), arrowprops=dict(arrowstyle="->", lw=1.6, color="#0072B2"))
    axA.text(5.00, 6.66, "Fix T_fit", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#0072B2")

    axA.annotate("", xy=(5.35, 4.86), xytext=(7.75, 5.42), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))
    axA.annotate("", xy=(4.65, 4.86), xytext=(2.25, 5.42), arrowprops=dict(arrowstyle="->", lw=1.2, linestyle="--", color="#555555"))

    draw_box(
        axA, 0.20, 2.94, 9.60, 1.84,
        "Held-Out Test Partition (D_test): N = 3,686 Images across K = 1,972 Grouping Units",
        "1,695 observed lesion_id clusters (max k=24 images) + 277 missing-ID singletons; NO patient_id column\n"
        "Gamma(x) = {c : 1 - p_T(c|x) <= q* = 0.904267}  [p_T >= 0.095733; 0.00% empty sets]",
        fc="#F5F5F7", ec="#34495E"
    )

    draw_box(
        axA, 0.20, 0.18, 9.60, 2.42,
        "Methodological Qualification: Disjointness vs. Exchangeability vs. Independence",
        "Group-disjoint partitioning prevents overlap of known lesion-ID groups between fitting and conformal\n"
        "subsets. It does not independently establish exchangeability or independence of all image-level\n"
        "observations, and patient-level independence cannot be verified from the available metadata.\n"
        "Note: Within D_conf (141/396 multi-image groups) & D_test (746/1,972), repeated views remain correlated.",
        fc="#FFF3EE", ec="#D55E00", lw=1.5, title_color="#9E3B00"
    )

    # -------------------------------------------------------------------------
    # PANEL B: Set Size Distribution & Empirical Marginal Coverage
    # -------------------------------------------------------------------------
    set_sizes = [0, 1, 2, 3, 4, 5, 6]
    counts_sz = [row_primary[f"size_{s}_count"] for s in set_sizes]
    pcts_sz = [row_primary[f"size_{s}_pct"] for s in set_sizes]
    bar_colors = ["#999999", "#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#CC79A7", "#7F7F7F"]

    bars = axB.bar(set_sizes, pcts_sz, width=0.66, color=bar_colors, edgecolor="#222222", linewidth=0.8, zorder=3)
    axB.set_ylim(0, 62)
    axB.set_xlim(-0.6, 6.65)
    axB.set_xticks(set_sizes)
    axB.set_xticklabels([f"{s}" for s in set_sizes])
    axB.set_xlabel("Prediction Set Size |Gamma(x)| (Classes out of 7)")
    axB.set_ylabel("Proportion of Held-Out Test Images (%)")
    axB.set_title("B. Prediction Set Size Distribution & Marginal Coverage (N = 3,686)", loc="left", pad=10)
    axB.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    for b, cnt, pct in zip(bars, counts_sz, pcts_sz):
        label_txt = f"{pct:.2f}%\n({cnt:,})" if cnt > 0 else "0.00%\n(0)"
        axB.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.9,
            label_txt,
            ha="center",
            va="bottom",
            fontsize=7.7,
            color="#111111",
            fontweight="bold" if cnt > 1000 else "normal",
        )

    ci_m_lo = row_primary["ci95_low"] * 100.0
    ci_m_hi = row_primary["ci95_high"] * 100.0
    sz_lo = row_primary["set_size_ci95_low"]
    sz_hi = row_primary["set_size_ci95_high"]
    panel_b_note = (
        "Observed empirical marginal image-level coverage was 90.67% on the held-out test set.\n"
        f"• Exact (q* rank 634 = 0.904267): 3,342 / 3,686 (95% cluster CI [{ci_m_lo:.2f}%, {ci_m_hi:.2f}%])\n"
        f"• Mean Set Size: 1.9249 (95% CI [{sz_lo:.3f}, {sz_hi:.3f}]), SD = 0.916  |  0.00% Empty (0)\n"
        f"• Singletons: 38.09% (1,404 / 3,686)  |  Full 7-class sets: 0.00% (0 / 3,686)\n"
        f"• 1,000 Group Splits (k = 634) [p2.5, p97.5]: {split_covs.mean()*100:.2f}% "
        f"[{np.percentile(split_covs,2.5)*100:.2f}%, {np.percentile(split_covs,97.5)*100:.2f}%]  |  Reused CAL: {row_reused['marginal_coverage']*100:.2f}%"
    )
    axB.text(
        0.02,
        0.975,
        panel_b_note,
        transform=axB.transAxes,
        ha="left",
        va="top",
        fontsize=7.25,
        color="#111111",
        linespacing=1.34,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#F8FAFC", edgecolor="#0072B2", linewidth=1.1),
    )

    # -------------------------------------------------------------------------
    # PANEL C: Class-Conditional Empirical Coverage (7 Classes, NO GREEN!)
    # -------------------------------------------------------------------------
    y_pos = np.arange(len(panel_c_df))
    cov_vals = panel_c_df["empirical_coverage"].values * 100.0
    ci_lows = panel_c_df["cluster_ci95_low"].values * 100.0
    ci_highs = panel_c_df["cluster_ci95_high"].values * 100.0
    w_lows = panel_c_df["wilson_ci95_low"].values * 100.0
    w_highs = panel_c_df["wilson_ci95_high"].values * 100.0
    codes_ord = panel_c_df["class_code"].values
    ns_ord = panel_c_df["support_n"].values
    covs_n = panel_c_df["covered_n"].values

    axC.axvline(90.0, color="#B22222", linestyle="--", linewidth=1.4, zorder=2)
    axC.axvline(row_primary["marginal_coverage"] * 100.0, color="#0072B2", linestyle=":", linewidth=1.3, zorder=2)

    for i, (code, val, lo, hi, wlo, whi, nc, cn) in enumerate(
        zip(codes_ord, cov_vals, ci_lows, ci_highs, w_lows, w_highs, ns_ord, covs_n)
    ):
        lc = code.lower()
        if lc == "vasc":
            col = "#5C6670"
            marker = "^"
            axC.plot([wlo, whi], [i, i], color="#5C6670", linestyle="--", linewidth=2.0, zorder=3)
            axC.plot([wlo, wlo], [i - 0.12, i + 0.12], color="#5C6670", linewidth=1.3, zorder=3)
            axC.plot([whi, whi], [i - 0.12, i + 0.12], color="#5C6670", linewidth=1.3, zorder=3)
            axC.scatter([val], [i], color=col, edgecolor="#111111", s=70, marker=marker, zorder=4)
            txt = f"100.00% (19/19)\n[Wilson CI {wlo:.1f}%–{whi:.1f}%; sparse]"
        else:
            if val < 80.0:
                col = "#D55E00"
                marker = "o"
            elif val < 90.0:
                col = "#E69F00"
                marker = "D"
            else:
                col = "#0072B2"
                marker = "s"
            axC.barh(i, val, height=0.52, color=col, alpha=0.22, edgecolor=col, linewidth=1.0, zorder=2)
            axC.errorbar(
                val, i,
                xerr=[[max(0.0, val - lo)], [max(0.0, hi - val)]],
                fmt=marker, color=col, markeredgecolor="#111111", markersize=7.0,
                ecolor="#222222", elinewidth=1.4, capsize=4, zorder=4,
            )
            txt = f"{val:.2f}% ({cn}/{nc}) [{lo:.1f}%, {hi:.1f}%]"

        axC.text(102.0, i, txt, va="center", ha="left", fontsize=7.6, color="#111111",
                 fontweight="bold" if lc == "akiec" else "normal")

    axC.set_yticks(y_pos)
    axC.set_yticklabels([SHORT_LABELS[c.lower()] for c in codes_ord], fontsize=8.3)
    axC.set_ylim(-1.55, 6.65)
    axC.set_xlim(0, 148)
    axC.set_xticks([0, 20, 40, 60, 80, 90, 100])
    axC.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "90%", "100%"])
    axC.set_xlabel("Empirical Image-Level Coverage (%) [95% CI & Exact Numerator/Denominator]")
    axC.set_title("C. Class-Conditional Empirical Coverage Disparity (N = 3,686)", loc="left", pad=10)
    axC.grid(axis="x", linestyle=":", alpha=0.55, zorder=0)
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    legend_handles_c = [
        Line2D([0], [0], color="#B22222", linestyle="--", lw=1.4, label="90.0% Nominal Target"),
        Line2D([0], [0], color="#0072B2", linestyle=":", lw=1.3, label="90.67% Empirical Marginal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D55E00", markeredgecolor="#111", markersize=6.8, label="< 80% Severe Under-Coverage"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E69F00", markeredgecolor="#111", markersize=6.2, label="80–90% Below Marginal Target"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#0072B2", markeredgecolor="#111", markersize=6.2, label=">= 90% Coverage (n >= 466)"),
        Line2D([0], [0], marker="^", color="#5C6670", linestyle="--", lw=1.5, markeredgecolor="#111", markersize=6.8, label="Sparse n=19 (Wilson CI; Not Stability)"),
    ]
    axC.legend(
        handles=legend_handles_c,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.015),
        fontsize=6.95,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        ncol=2,
        columnspacing=0.9,
        handlelength=1.8,
    )

    # -------------------------------------------------------------------------
    # PANEL D: Cluster-Stratified & Descriptive Lesion-Level Coverage (K = 1,972)
    # -------------------------------------------------------------------------
    d_lookup = {r["metric_id"]: r for r in panel_d_rows}
    selected_ids = [
        "grp_all_cov_multi",
        "img_multi_grp",
        "grp_all_cov",
        "img_all",
        "grp_any_cov",
        "img_singleton_grp",
    ]
    selected_labels = [
        "Cluster All-Covered\n(Multi-Img, K=746)",
        "Image-Level Cov.\n(Multi-Img, N=2,460)",
        "Cluster All-Covered\n(All Groups, K=1,972)",
        "Image-Level Marginal\n(All Images, N=3,686)",
        "Cluster Any-Covered\n(All Groups, K=1,972)",
        "Singleton Groups\n(k=1, N=K=1,226)",
    ]
    selected_colors = ["#D55E00", "#E69F00", "#CC79A7", "#0072B2", "#4D648D", "#56B4E9"]
    selected_rows = [d_lookup[mid] for mid in selected_ids]

    yd = np.array([0.0, 1.0, 2.0, 3.7, 4.7, 5.7])
    axD.axvline(90.0, color="#B22222", linestyle="--", linewidth=1.4, zorder=2)

    for pos, r, lbl, col in zip(yd, selected_rows, selected_labels, selected_colors):
        val = r["empirical_coverage"] * 100.0
        lo = r["cluster_ci95_low"] * 100.0
        hi = r["cluster_ci95_high"] * 100.0
        axD.barh(pos, val, height=0.56, color=col, alpha=0.82, edgecolor="#222222", linewidth=0.8, zorder=3)
        axD.errorbar(
            val, pos,
            xerr=[[max(0.0, val - lo)], [max(0.0, hi - val)]],
            fmt="none", ecolor="#111111", elinewidth=1.4, capsize=4, zorder=4,
        )
        txt = f"{val:.2f}% ({r['numerator']:,}/{r['denominator']:,}) [{lo:.1f}%, {hi:.1f}%]"
        axD.text(101.8, pos, txt, va="center", ha="left", fontsize=7.6, color="#111111")

    axD.axhline(2.85, color="#777777", linestyle="-.", linewidth=0.95, alpha=0.8)
    axD.text(
        2.0,
        2.95,
        "Above: Aggregate & Singleton References   |   Below: Multi-Image & Strict Cluster-All Criteria",
        fontsize=7.15,
        color="#333333",
        style="italic",
        va="bottom",
    )

    axD.set_yticks(yd)
    axD.set_yticklabels(selected_labels, fontsize=8.0)
    axD.set_ylim(-1.15, 6.35)
    axD.set_xlim(0, 148)
    axD.set_xticks([0, 20, 40, 60, 80, 90, 100])
    axD.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "90%", "100%"])
    axD.set_xlabel("Empirical Coverage (%) [95% Cluster Bootstrap CI, B=2,000] & Numerator/Denominator")
    axD.set_title("D. Descriptive Cluster Coverage & Multiplicity Strata (K = 1,972)", loc="left", pad=10)
    axD.grid(axis="x", linestyle=":", alpha=0.55, zorder=0)
    axD.spines["top"].set_visible(False)
    axD.spines["right"].set_visible(False)

    legend_handles_d = [
        Line2D([0], [0], color="#B22222", linestyle="--", lw=1.4, label="90.0% Nominal Image Target"),
        mpatches.Patch(facecolor="#D55E00", edgecolor="#222", alpha=0.82, label="Below 85% (Multi-Img Cluster-All)"),
        mpatches.Patch(facecolor="#0072B2", edgecolor="#222", alpha=0.82, label="Image-Level Marginal (90.67%)"),
    ]
    axD.legend(
        handles=legend_handles_d,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.015),
        fontsize=7.0,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        ncol=3,
        columnspacing=0.9,
        handlelength=1.5,
    )

    footnote_d = (
        "Footnote: Empirical coverage is reported on the held-out image set. "
        "Repeated images per lesion and incomplete patient identifiers limit strict exchangeability claims."
    )
    fig.text(0.50, 0.014, footnote_d, ha="center", va="bottom", fontsize=8.3, style="italic", color="#333333",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#CCCCCC", linewidth=0.8))

    fig.subplots_adjust(left=0.105, right=0.98, top=0.92, bottom=0.095, wspace=0.45, hspace=0.32)

    for ext in ("pdf", "svg", "png"):
        out_path = OUT_DIR / f"fig7_conformal_uncertainty.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
