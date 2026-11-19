"""
DERMA-ACT V2 — Figure 8 Source Script
Controlled Digital Perturbation Robustness & Predictive Distribution Sensitivity

Strictly post-hoc, reproducible analysis from frozen validation perturbation logs.
Zero model training, fine-tuning, or classifier inference is performed.
"""

from __future__ import annotations

import json

import argparse as _fg_argparse
from pathlib import Path as _FgPath
_fg_ap = _fg_argparse.ArgumentParser(description='Figure 8: Digital Perturbation Sensitivity (Tier B robustness JSON & splits required)')
_fg_ap.add_argument('--root', type=_FgPath, default=_FgPath(__file__).resolve().parents[2], help='Data/repository root')
_fg_ap.add_argument('--out-dir', type=_FgPath, default=_FgPath(__file__).resolve().parents[2] / 'outputs/figures', help='Output figure directory')
_fg_args, _ = _fg_ap.parse_known_args()
_FG_ROOT = _fg_args.root.resolve()
_FG_OUT = _fg_args.out_dir.resolve()
_FG_OUT.mkdir(parents=True, exist_ok=True)

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ==============================================================================
# 1. Paths & Global Styling (Okabe-Ito Colorblind-Safe)
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _FG_ROOT

_pc_root = REPO_ROOT / "PROJECT_COMPLETION"
if not _pc_root.exists():
    _pc_archive = REPO_ROOT.parent / f"{REPO_ROOT.name}_ARCHIVE" / "HISTORICAL_PROJECTS" / "PROJECT_COMPLETION"
    if _pc_archive.exists():
        _pc_root = _pc_archive

RESULTS_PATH = _pc_root / "ROBUSTNESS" / "ROBUSTNESS_RESULTS.json"
MANIFEST_PATH = _pc_root / "ROBUSTNESS" / "ROBUSTNESS_COHORT_MANIFEST.json"
CORRECTED_CI_PATH = _pc_root / "FINAL_AUDIT" / "ROBUSTNESS_CORRECTED_CI_RESULTS.json"
VAL_SPLIT_PATH = REPO_ROOT / "data" / "isic2019" / "splits" / "val_indices.csv"
CLEANED_CSV_PATH = REPO_ROOT / "data" / "isic2019" / "labels" / "cleaned.csv"

CLASSES_ORDER = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLS_TO_ID = {"mel": 0, "nv": 1, "bcc": 2, "akiec": 3, "bkl": 4, "df": 5, "vasc": 6}

# Okabe-Ito Palette
COLOR_SPATIAL = "#D55E00"      # Vermilion
COLOR_DEGRAD = "#0072B2"       # Blue (Compression / Blur / Noise)
COLOR_PHOTOM = "#009E73"       # Bluish Green
COLOR_IDENTITY = "#333333"     # Dark Charcoal

COLOR_EST_A = "#D55E00"        # Vermilion
COLOR_EST_B = "#0072B2"        # Blue
COLOR_EST_C = "#CC79A7"        # Reddish Purple

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.8,
    "figure.titlesize": 12.0,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def format_condition(ptype: str, params: dict) -> Tuple[str, str, str]:
    """Return (condition_id, human_label, category)."""
    if ptype == "identity":
        return "identity", "Identity (Unperturbed)", "Identity Control"
    if ptype == "jpeg":
        q = params["quality"]
        return f"jpeg_q{q}", f"JPEG (quality={q})", "Optical / Compression"
    if ptype == "brightness":
        f = params["factor"]
        pct = "+5%" if f > 1.0 else "-5%"
        return f"brightness_{f:.2f}", f"Brightness ({pct}, x{f:.2f})", "Photometric"
    if ptype == "contrast":
        f = params["factor"]
        pct = "+5%" if f > 1.0 else "-5%"
        return f"contrast_{f:.2f}", f"Contrast ({pct}, x{f:.2f})", "Photometric"
    if ptype == "gamma":
        g = params["gamma"]
        return f"gamma_{g:.2f}", f"Gamma ($\\gamma={g:.2f}$)", "Photometric"
    if ptype == "blur":
        r = params["radius"]
        return f"blur_r{r:.1f}", f"Gaussian Blur ($r={r:.1f}$ px)", "Optical / Compression"
    if ptype == "noise":
        s = params["std"]
        return f"noise_std{s:.2f}", f"Gaussian Noise ($\\sigma={s:.2f}$)", "Optical / Compression"
    if ptype == "translation":
        tx, ty = params["tx"], params["ty"]
        sign_lbl = f"+{tx},+{ty}" if tx > 0 else f"{tx},{ty}"
        return f"trans_{tx}_{ty}", f"Translation ({sign_lbl} px)", "Spatial / Geometric"
    if ptype == "resize":
        sc = params["scale"]
        pct = "+5%" if sc > 1.0 else "-5%"
        return f"resize_{sc:.2f}", f"Resize ({pct}, {sc:.2f}x)", "Spatial / Geometric"
    if ptype == "crop":
        pct = int(round(params["percent"] * 100))
        return f"crop_{pct}pct", f"Center Crop ({pct}% margin)", "Spatial / Geometric"
    return f"{ptype}", f"{ptype} {params}", "Other"


def load_and_compute() -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Load frozen robustness artifacts and compute unified multi-estimand metrics."""
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(CORRECTED_CI_PATH, "r", encoding="utf-8") as f:
        corrected = json.load(f)

    val_ids = pd.read_csv(VAL_SPLIT_PATH, header=None)[0].tolist()
    cleaned_df = pd.read_csv(CLEANED_CSV_PATH)
    val_meta = cleaned_df[cleaned_df["image_id"].isin(val_ids)]
    val_counts = val_meta["class_name"].value_counts().to_dict()
    n_val_total = sum(val_counts.values())
    assert n_val_total == 3542, f"Expected N_val=3542, got {n_val_total}"
    p_val = {c: val_counts[c] / float(n_val_total) for c in CLASSES_ORDER}

    cases = results["detailed_case_results"]
    n_cases = len(cases)
    assert n_cases == 50, f"Expected N=50 cases, got {n_cases}"
    gt_cls = [c["ground_truth_class"] for c in cases]
    lesion_ids = [m["lesion_id"] for m in manifest]
    mask_47 = np.array([str(lid).lower() not in ("nan", "none", "") for lid in lesion_ids])
    assert int(mask_47.sum()) == 47, f"Expected 47 known lesions, got {mask_47.sum()}"

    base_preds = np.array([c["base_prediction"] for c in cases])
    gt_ids = np.array([CLS_TO_ID[c["ground_truth_class"]] for c in cases])
    base_correct = (base_preds == gt_ids)

    # Paired Stratified Image-Level Bootstrap (B=1,000, seed=20261112)
    strata_idx = {c: [i for i, g in enumerate(gt_cls) if g == c] for c in CLASSES_ORDER}
    rng = np.random.default_rng(20261112)
    B = 1000
    boot_indices = np.zeros((B, n_cases), dtype=int)
    for b in range(B):
        row: List[int] = []
        for c in CLASSES_ORDER:
            idxs = strata_idx[c]
            row.extend(rng.choice(idxs, size=len(idxs), replace=True).tolist())
        boot_indices[b] = row

    stratum_slices: Dict[str, slice] = {}
    start = 0
    for c in CLASSES_ORDER:
        end = start + len(strata_idx[c])
        stratum_slices[c] = slice(start, end)
        start = end

    def estimand_stats(arr: np.ndarray) -> dict:
        est_a = float(np.mean(arr))
        cls_means = {c: float(np.mean(arr[strata_idx[c]])) for c in CLASSES_ORDER}
        est_b = float(sum(p_val[c] * cls_means[c] for c in CLASSES_ORDER))
        est_c = float(np.mean([cls_means[c] for c in CLASSES_ORDER]))

        b_vals = arr[boot_indices]  # (B, 50)
        b_a = b_vals.mean(axis=1)
        b_cls = {c: b_vals[:, stratum_slices[c]].mean(axis=1) for c in CLASSES_ORDER}
        b_b = sum(p_val[c] * b_cls[c] for c in CLASSES_ORDER)
        b_c = sum((1.0 / 7.0) * b_cls[c] for c in CLASSES_ORDER)

        return {
            "A": est_a,
            "A_lo": float(np.percentile(b_a, 2.5)),
            "A_hi": float(np.percentile(b_a, 97.5)),
            "B": est_b,
            "B_lo": float(np.percentile(b_b, 2.5)),
            "B_hi": float(np.percentile(b_b, 97.5)),
            "C": est_c,
            "C_lo": float(np.percentile(b_c, 2.5)),
            "C_hi": float(np.percentile(b_c, 97.5)),
            "b_a": b_a,
            "cls_means": cls_means,
        }

    rows = []
    for v_idx in range(16):
        p0 = cases[0]["perturbations"][v_idx]
        pt, prm = p0["perturbation_type"], p0["parameters"]
        cond_id, label, category = format_condition(pt, prm)

        con = np.array([1.0 if c["perturbations"][v_idx]["prediction_consistency"] else 0.0 for c in cases])
        tvd = np.array([c["perturbations"][v_idx]["tvd"] for c in cases])
        ent = np.array([c["perturbations"][v_idx]["entropy_shift"] for c in cases])
        mabs = np.array([c["perturbations"][v_idx]["max_abs_deviation"] for c in cases])
        mrel = np.array([c["perturbations"][v_idx]["max_rel_deviation"] for c in cases])
        fail = np.array([1.0 if c["perturbations"][v_idx]["is_failure"] else 0.0 for c in cases])

        flips = con == 0.0
        n_flips = int(flips.sum())
        n_flips_corr = int((flips & base_correct).sum())
        n_flips_wrong = int((flips & ~base_correct).sum())
        n_flips_vasc = int(sum(1 for i in strata_idx["vasc"] if flips[i]))

        s_con = estimand_stats(con)
        s_tvd = estimand_stats(tvd)
        s_ent = estimand_stats(ent)

        # Cross-verify point estimates against ROBUSTNESS_CORRECTED_CI_RESULTS.json
        cv = corrected["variant_results"][v_idx]
        assert abs(s_con["A"] - cv["estimand_A_cohort_descriptive"]["top1_consistency"]["mean"]) < 1e-12
        assert abs(s_con["B"] - cv["estimand_B_validation_prevalence_weighted"]["top1_consistency"]["mean"]) < 1e-12
        assert abs(s_con["C"] - cv["estimand_C_class_macro_average"]["top1_consistency"]["mean"]) < 1e-12
        assert abs(s_tvd["A"] - cv["estimand_A_cohort_descriptive"]["total_variation_distance"]["mean"]) < 1e-12
        assert abs(s_tvd["B"] - cv["estimand_B_validation_prevalence_weighted"]["total_variation_distance"]["mean"]) < 1e-12
        assert abs(s_tvd["C"] - cv["estimand_C_class_macro_average"]["total_variation_distance"]["mean"]) < 1e-12
        assert abs(s_ent["A"] - cv["estimand_A_cohort_descriptive"]["entropy_shift"]["mean"]) < 1e-12

        rows.append({
            "variant_index": v_idx,
            "condition_id": cond_id,
            "label": label,
            "perturbation_family": pt,
            "parameters": json.dumps(prm, sort_keys=True),
            "category": category,
            "n_images": n_cases,
            "n_flips": n_flips,
            "flips_on_baseline_correct": n_flips_corr,
            "flips_on_baseline_wrong": n_flips_wrong,
            "flips_vasc_n2": n_flips_vasc,
            # Top-1 Consistency
            "con_A": s_con["A"], "con_A_lo": s_con["A_lo"], "con_A_hi": s_con["A_hi"],
            "con_B": s_con["B"], "con_B_lo": s_con["B_lo"], "con_B_hi": s_con["B_hi"],
            "con_C": s_con["C"], "con_C_lo": s_con["C_lo"], "con_C_hi": s_con["C_hi"],
            "con_N47": float(con[mask_47].mean()),
            # TVD
            "tvd_A": s_tvd["A"], "tvd_A_lo": s_tvd["A_lo"], "tvd_A_hi": s_tvd["A_hi"],
            "tvd_B": s_tvd["B"], "tvd_B_lo": s_tvd["B_lo"], "tvd_B_hi": s_tvd["B_hi"],
            "tvd_C": s_tvd["C"], "tvd_C_lo": s_tvd["C_lo"], "tvd_C_hi": s_tvd["C_hi"],
            "tvd_N47": float(tvd[mask_47].mean()),
            "tvd_median": float(np.median(tvd)),
            "tvd_q25": float(np.percentile(tvd, 25)),
            "tvd_q75": float(np.percentile(tvd, 75)),
            "tvd_iqr": float(np.percentile(tvd, 75) - np.percentile(tvd, 25)),
            "tvd_max": float(np.max(tvd)),
            # Absolute Entropy Shift |dH| (nats)
            "abs_dH_A": s_ent["A"], "abs_dH_A_lo": s_ent["A_lo"], "abs_dH_A_hi": s_ent["A_hi"],
            "abs_dH_B": s_ent["B"], "abs_dH_B_lo": s_ent["B_lo"], "abs_dH_B_hi": s_ent["B_hi"],
            "abs_dH_C": s_ent["C"], "abs_dH_C_lo": s_ent["C_lo"], "abs_dH_C_hi": s_ent["C_hi"],
            "abs_dH_N47": float(ent[mask_47].mean()),
            "abs_dH_median": float(np.median(ent)),
            "abs_dH_q25": float(np.percentile(ent, 25)),
            "abs_dH_q75": float(np.percentile(ent, 75)),
            "abs_dH_iqr": float(np.percentile(ent, 75) - np.percentile(ent, 25)),
            "abs_dH_max": float(np.max(ent)),
            # Auxiliary Audit Context (semantic relative/abs dev & legacy composite flag)
            "semantic_abs_dev_mean": float(mabs.mean()),
            "semantic_abs_dev_median": float(np.median(mabs)),
            "semantic_rel_dev_mean_valid": float(mrel.mean()),
            "semantic_rel_dev_median_valid": float(np.median(mrel)),
            "composite_alert_rate": float(fail.mean()),
        })

    df = pd.DataFrame(rows)

    # Ranks over 15 non-identity perturbations
    mask_nonid = df["perturbation_family"] != "identity"
    df["rank_tvd_A"] = np.nan
    df["rank_tvd_B"] = np.nan
    df["rank_tvd_C"] = np.nan
    df["rank_con_worst_A"] = np.nan
    df["rank_con_worst_B"] = np.nan
    df["rank_con_worst_C"] = np.nan
    df.loc[mask_nonid, "rank_tvd_A"] = df.loc[mask_nonid, "tvd_A"].rank(ascending=False, method="min").astype(int)
    df.loc[mask_nonid, "rank_tvd_B"] = df.loc[mask_nonid, "tvd_B"].rank(ascending=False, method="min").astype(int)
    df.loc[mask_nonid, "rank_tvd_C"] = df.loc[mask_nonid, "tvd_C"].rank(ascending=False, method="min").astype(int)
    df.loc[mask_nonid, "rank_con_worst_A"] = df.loc[mask_nonid, "con_A"].rank(ascending=True, method="min").astype(int)
    df.loc[mask_nonid, "rank_con_worst_B"] = df.loc[mask_nonid, "con_B"].rank(ascending=True, method="min").astype(int)
    df.loc[mask_nonid, "rank_con_worst_C"] = df.loc[mask_nonid, "con_C"].rank(ascending=True, method="min").astype(int)

    # Paired contrasts (e.g., Brightness 0.95 vs 1.05, JPEG 75 vs 95)
    def paired_diff(idx1: int, idx2: int) -> dict:
        t1 = np.array([c["perturbations"][idx1]["tvd"] for c in cases])
        t2 = np.array([c["perturbations"][idx2]["tvd"] for c in cases])
        d = t1 - t2
        bd = d[boot_indices].mean(axis=1)
        return {
            "diff_mean": float(d.mean()),
            "diff_ci_lo": float(np.percentile(bd, 2.5)),
            "diff_ci_hi": float(np.percentile(bd, 97.5)),
            "ratio": float(t1.mean() / t2.mean()),
        }

    meta_stats = {
        "n_val_total": n_val_total,
        "val_counts": val_counts,
        "base_accuracy_n50": float(base_correct.mean()),
        "base_correct_count": int(base_correct.sum()),
        "brightness_asym": paired_diff(2, 3),  # 0.95 vs 1.05
        "jpeg_contrast": paired_diff(1, 0),    # 75 vs 95
        "max_diff_N47_tvd": float(np.max(np.abs(df["tvd_N47"] - df["tvd_A"]))),
        "max_diff_N47_con": float(np.max(np.abs(df["con_N47"] - df["con_A"]))),
    }
    return df, meta_stats


def export_csvs(df: pd.DataFrame) -> None:
    """Export master and panel-specific CSV tables cleanly."""
    df.to_csv(SCRIPT_DIR / "FIGURE_8_PERTURBATION_DATA.csv", index=False)

    panel_a_cols = [
        "variant_index", "condition_id", "label", "category", "n_images",
        "con_A", "con_A_lo", "con_A_hi", "n_flips",
        "flips_on_baseline_correct", "flips_on_baseline_wrong", "flips_vasc_n2",
        "con_N47"
    ]
    df.sort_values(["con_A", "tvd_A"], ascending=[True, False])[panel_a_cols].to_csv(
        SCRIPT_DIR / "FIGURE_8_PANEL_A_CONSISTENCY.csv", index=False
    )

    panel_b_cols = [
        "variant_index", "condition_id", "label", "category", "n_images",
        "tvd_A", "tvd_A_lo", "tvd_A_hi", "tvd_median", "tvd_q25", "tvd_q75",
        "tvd_iqr", "tvd_max", "tvd_N47", "rank_tvd_A"
    ]
    df.sort_values("tvd_A", ascending=False)[panel_b_cols].to_csv(
        SCRIPT_DIR / "FIGURE_8_PANEL_B_TVD.csv", index=False
    )

    panel_c_cols = [
        "variant_index", "condition_id", "label", "category", "n_images",
        "abs_dH_A", "abs_dH_A_lo", "abs_dH_A_hi", "abs_dH_median", "abs_dH_q25",
        "abs_dH_q75", "abs_dH_iqr", "abs_dH_max", "abs_dH_N47"
    ]
    df.sort_values("tvd_A", ascending=False)[panel_c_cols].to_csv(
        SCRIPT_DIR / "FIGURE_8_PANEL_C_ENTROPY.csv", index=False
    )

    panel_d_cols = [
        "variant_index", "condition_id", "label", "category",
        "tvd_A", "tvd_A_lo", "tvd_A_hi", "rank_tvd_A",
        "tvd_B", "tvd_B_lo", "tvd_B_hi", "rank_tvd_B",
        "tvd_C", "tvd_C_lo", "tvd_C_hi", "rank_tvd_C",
        "con_A", "rank_con_worst_A", "con_B", "rank_con_worst_B", "con_C", "rank_con_worst_C",
        "abs_dH_A", "abs_dH_B", "abs_dH_C"
    ]
    df[df["perturbation_family"] != "identity"].sort_values("tvd_A", ascending=False)[panel_d_cols].to_csv(
        SCRIPT_DIR / "FIGURE_8_PANEL_D_ESTIMAND_SENSITIVITY.csv", index=False
    )


def cat_style(category: str) -> Tuple[str, str]:
    if category == "Spatial / Geometric":
        return COLOR_SPATIAL, "s"
    if category == "Optical / Compression":
        return COLOR_DEGRAD, "o"
    if category == "Photometric":
        return COLOR_PHOTOM, "^"
    return COLOR_IDENTITY, "D"


def render_figure(df: pd.DataFrame, meta: dict) -> None:
    """Render 4-panel publication figure to PDF, SVG, and 300 DPI PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 11.6))
    plt.subplots_adjust(left=0.14, right=0.975, top=0.905, bottom=0.095, wspace=0.43, hspace=0.28)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # --------------------------------------------------------------------------
    # PANEL A: Top-1 Prediction Consistency (%) — Dot + 95% Paired Stratified CI
    # --------------------------------------------------------------------------
    df_a = df.sort_values(["con_A", "tvd_A"], ascending=[True, False]).reset_index(drop=True)
    y_pos = np.arange(len(df_a))
    ax_a.axvline(100.0, color=COLOR_IDENTITY, linestyle="--", linewidth=1.2, alpha=0.75, zorder=1)

    for i, r in df_a.iterrows():
        col, mkr = cat_style(r["category"])
        mean_pct = r["con_A"] * 100.0
        lo_pct = r["con_A_lo"] * 100.0
        hi_pct = r["con_A_hi"] * 100.0
        ax_a.plot([lo_pct, hi_pct], [i, i], color=col, linewidth=2.1, alpha=0.88, zorder=2)
        ax_a.plot([lo_pct, lo_pct], [i - 0.14, i + 0.14], color=col, linewidth=1.3, zorder=2)
        ax_a.plot([hi_pct, hi_pct], [i - 0.14, i + 0.14], color=col, linewidth=1.3, zorder=2)
        ax_a.scatter([mean_pct], [i], color=col, marker=mkr, s=46, edgecolor="white", linewidth=0.7, zorder=4)
        flip_txt = f"{mean_pct:5.1f}% ({int(r['n_flips'])}/50)" if r["n_flips"] > 0 else "100.0% (0/50)"
        ax_a.text(100.9, i, flip_txt, va="center", ha="left", fontsize=7.6, color="#222222")

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(df_a["label"], fontsize=8.0)
    ax_a.invert_yaxis()
    ax_a.set_xlim(79.5, 110.5)
    ax_a.set_xticks([80, 85, 90, 95, 100])
    ax_a.set_xticklabels(["80%", "85%", "90%", "95%", "100%"])
    ax_a.set_xlabel("Top-1 Prediction Consistency wrt Baseline (%) [Dot & 95% CI]")
    ax_a.set_title("A. Top-1 Prediction Consistency Across 16 Conditions (N=50)", loc="left", pad=8)
    ax_a.grid(axis="x", linestyle=":", alpha=0.45)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    ax_a.text(
        0.025, 0.035,
        f"Note: Consistency $\\neq$ accuracy\n(Baseline N=50 acc: {meta['base_accuracy_n50']*100:.1f}%, {meta['base_correct_count']}/50).\n"
        "Values in parens: flips / 50.",
        transform=ax_a.transAxes, fontsize=7.4, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.94),
    )

    # --------------------------------------------------------------------------
    # Shared Ordering for Panels B & C: Sorted descending by Estimand A Mean TVD
    # --------------------------------------------------------------------------
    df_bc = df.sort_values("tvd_A", ascending=False).reset_index(drop=True)
    y_bc = np.arange(len(df_bc))

    # --------------------------------------------------------------------------
    # PANEL B: Total Variation Distance (TVD)
    # --------------------------------------------------------------------------
    ax_b.axvline(0.0, color=COLOR_IDENTITY, linestyle="--", linewidth=1.1, alpha=0.7, zorder=1)
    for i, r in df_bc.iterrows():
        col, mkr = cat_style(r["category"])
        ax_b.barh(i, r["tvd_A"], height=0.62, color=col, alpha=0.36, edgecolor=col, linewidth=0.9, zorder=2)
        ax_b.plot([r["tvd_A_lo"], r["tvd_A_hi"]], [i, i], color=col, linewidth=2.0, zorder=3)
        ax_b.scatter([r["tvd_median"]], [i], color="white", edgecolor=col, marker="D", s=26, linewidth=1.3, zorder=4)
        x_txt = max(r["tvd_A_hi"] + 0.0028, 0.0045)
        ax_b.text(
            x_txt, i,
            f"{r['tvd_A']:.4f} [{r['tvd_A_lo']:.3f}, {r['tvd_A_hi']:.3f}]",
            va="center", ha="left", fontsize=7.3, color="#222222"
        )

    ax_b.set_yticks(y_bc)
    ax_b.set_yticklabels(df_bc["label"], fontsize=8.0)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.0, 0.155)
    ax_b.set_xlabel("Total Variation Distance, $\\mathrm{TVD} = \\frac{1}{2} \\|\\tilde{p} - p\\|_1$")
    ax_b.set_title("B. Distributional Drift (Mean TVD, 95% CI & Median)", loc="left", pad=8)
    ax_b.grid(axis="x", linestyle=":", alpha=0.45)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    ba = meta["brightness_asym"]
    ax_b.text(
        0.975, 0.13,
        f"Brightness Asymmetry (-5% vs +5%):\n"
        f"+{(ba['ratio']-1)*100:.1f}% larger TVD ($\\Delta$={ba['diff_mean']:+.4f},\n"
        f"paired 95% CI [{ba['diff_ci_lo']:+.4f}, {ba['diff_ci_hi']:+.4f}])\n\n"
        "Spatial transforms (crop/shift) induce\n"
        "largest probability drift (TVD 0.078-0.086).",
        transform=ax_b.transAxes, fontsize=7.3, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8F4", edgecolor="#D55E00", alpha=0.94),
    )

    # --------------------------------------------------------------------------
    # PANEL C: Absolute Predictive Entropy Shift |dH| (nats)
    # --------------------------------------------------------------------------
    ax_c.axvline(0.0, color=COLOR_IDENTITY, linestyle="--", linewidth=1.1, alpha=0.7, zorder=1)
    for i, r in df_bc.iterrows():
        col, mkr = cat_style(r["category"])
        ax_c.barh(i, r["abs_dH_A"], height=0.62, color=col, alpha=0.36, edgecolor=col, linewidth=0.9, zorder=2)
        ax_c.plot([r["abs_dH_A_lo"], r["abs_dH_A_hi"]], [i, i], color=col, linewidth=2.0, zorder=3)
        ax_c.scatter([r["abs_dH_median"]], [i], color="white", edgecolor=col, marker="D", s=26, linewidth=1.3, zorder=4)
        x_txt = max(r["abs_dH_A_hi"] + 0.0038, 0.0068)
        ax_c.text(
            x_txt, i,
            f"{r['abs_dH_A']:.4f} [{r['abs_dH_A_lo']:.3f}, {r['abs_dH_A_hi']:.3f}]",
            va="center", ha="left", fontsize=7.3, color="#222222"
        )

    ax_c.set_yticks(y_bc)
    ax_c.set_yticklabels(df_bc["label"], fontsize=8.0)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0.0, 0.235)
    ax_c.set_xlabel("Absolute Predictive Entropy Shift, $|\\Delta H| = |H(p) - H(\\tilde{p})|$ (nats)")
    ax_c.set_title("C. Predictive Entropy Sensitivity ($|\\Delta H|$, Shared Row Order)", loc="left", pad=8)
    ax_c.grid(axis="x", linestyle=":", alpha=0.45)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    ax_c.text(
        0.975, 0.10,
        "Artifact Scope & Caution:\n"
        "Only unsigned $|\\Delta H|$ is stored in frozen logs\n"
        "(raw probabilities not persisted; signed $\\Delta H$\n"
        "unreconstructable without inference).\n"
        "$|\\Delta H| \\neq$ calibrated clinical uncertainty.",
        transform=ax_c.transAxes, fontsize=7.3, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F8FB", edgecolor="#0072B2", alpha=0.94),
    )

    # --------------------------------------------------------------------------
    # PANEL D: Multi-Estimand Sensitivity (A vs B vs C) on Mean TVD
    # --------------------------------------------------------------------------
    df_d = df[df["perturbation_family"] != "identity"].sort_values("tvd_A", ascending=False).reset_index(drop=True)
    y_d = np.arange(len(df_d))
    ax_d.axvline(0.0, color=COLOR_IDENTITY, linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)

    offsets = [-0.22, 0.0, 0.22]
    est_specs = [
        ("tvd_A", "tvd_A_lo", "tvd_A_hi", COLOR_EST_A, "o", "Estimand A: Cohort Sample (N=50)"),
        ("tvd_B", "tvd_B_lo", "tvd_B_hi", COLOR_EST_B, "s", "Estimand B: Val-Prev Weighted ($N_{\\mathrm{val}}$=3,542 weights on N=50)"),
        ("tvd_C", "tvd_C_lo", "tvd_C_hi", COLOR_EST_C, "D", "Estimand C: Class-Macro Avg (1/7 per class)"),
    ]

    for off, (mcol, lcol, hcol, col, mkr, lbl) in zip(offsets, est_specs):
        ax_d.errorbar(
            df_d[mcol], y_d + off,
            xerr=[df_d[mcol] - df_d[lcol], df_d[hcol] - df_d[mcol]],
            fmt=mkr, color=col, ecolor=col, elinewidth=1.3, capsize=2.0,
            markersize=4.8, markeredgecolor="white", markeredgewidth=0.5,
            label=lbl, zorder=3
        )

    # Subtle row shading to aid comparison
    for i in range(len(df_d)):
        if i % 2 == 1:
            ax_d.axhspan(i - 0.48, i + 0.48, color="#F7F7F7", zorder=0)

    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(df_d["label"], fontsize=8.0)
    ax_d.invert_yaxis()
    ax_d.set_xlim(0.0, 0.125)
    ax_d.set_xlabel("Mean Total Variation Distance, TVD (95% Paired Bootstrap CI)")
    ax_d.set_title("D. Multi-Estimand Sensitivity (A vs B vs C)", loc="left", pad=8)
    ax_d.grid(axis="x", linestyle=":", alpha=0.45)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.legend(loc="lower right", bbox_to_anchor=(0.99, 0.30), frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=7.2)

    ax_d.text(
        0.985, 0.03,
        "Estimand & Small-Stratum Limits:\n"
        "- Est B reweights N=50 (not 3,542-img eval).\n"
        "- TVD rank swap: Trans (-5,-5) #1 in Est C\n"
        "  (0.0817) vs Crop 5% #1 in A/B (0.0862/0.0810).\n"
        "- Top-1 rank shift: single vasc (n=2) flip\n"
        "  drops Est C by -7.14 pp on 10 conditions;\n"
        "  DF (n=2) & VASC (n=2) strictly exploratory.",
        transform=ax_d.transAxes, fontsize=7.1, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FAF5FA", edgecolor="#CC79A7", alpha=0.92),
    )

    # Shared Family Legend at top
    legend_handles = [
        mlines.Line2D([], [], color=COLOR_SPATIAL, marker="s", linestyle="None", markersize=6.5, label="Spatial / Geometric"),
        mlines.Line2D([], [], color=COLOR_DEGRAD, marker="o", linestyle="None", markersize=6.5, label="Optical / Compression"),
        mlines.Line2D([], [], color=COLOR_PHOTOM, marker="^", linestyle="None", markersize=6.5, label="Photometric"),
        mlines.Line2D([], [], color=COLOR_IDENTITY, marker="D", linestyle="None", markersize=6.0, label="Identity Baseline (100% / 0.0)"),
        mlines.Line2D([], [], color="white", marker="D", markeredgecolor="#444444", markeredgewidth=1.2, linestyle="None", markersize=5.5, label="Panels B/C: Sample Median"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.55, 0.946),
        ncol=5, frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=8.0
    )

    fig.suptitle(
        "DERMA-ACT: Controlled Digital Perturbation Sensitivity on Validation Cohort (N = 50 Images, 800 Executions)",
        fontsize=12.2, fontweight="bold", y=0.982
    )
    fig.text(
        0.5, 0.014,
        "Cohort: 50 ISIC 2019 validation images (>=47 unique lesions, 3 unannotated; 0 overlap with test split N=3,686). "
        "Uncertainty bounds: paired stratified image-level percentile bootstrap (B=1,000, seed=20261112).\n"
        "Caveat: Synthetic Pillow transforms probe algorithmic sensitivity only and do not establish physical optical invariance, "
        "clinical camera-domain generalization, or calibrated clinical uncertainty.",
        ha="center", va="bottom", fontsize=7.6, color="#333333", linespacing=1.35
    )

    for ext in ("pdf", "svg", "png"):
        out_path = _FG_OUT / f"fig8_perturbation_robustness.{ext}"
        fig.savefig(out_path, dpi=300 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    df, meta = load_and_compute()
    # csv export skipped in figure dir
    render_figure(df, meta)
    print(f"Generated Figure 8 artifacts in {SCRIPT_DIR}")
    print(f"Verified N=50 validation cohort (baseline acc={meta['base_accuracy_n50']:.4f}), max |N47-N50| TVD={meta['max_diff_N47_tvd']:.5f}")


if __name__ == "__main__":
    main()
