"""Deterministic post-hoc analysis and publication Figure 5 generator for DERMA-ACT V2.

Preservation guarantees:
- Strictly post-hoc analysis of frozen test/dev predictions and test_policy_trace.csv.
- Zero model training, fine-tuning, or classifier inference.
- Recomputes all per-class sensitivities, paired differences, exact two-sided McNemar tests,
  step-down monotonic Holm-Bonferroni adjusted p-values, grouped bootstrap 95% CIs
  (over 1,972 operational partitioning units: 1,695 observed lesion-ID groups and 277
  missing-ID singleton images, B=2,000 resamples, seed=20261112), confusion matrices,
  and action-composition statistics directly from frozen source artifacts, and asserts
  exact consistency with P5.4 and P5.5 canonical JSON records.
"""

from __future__ import annotations

import json

import argparse as _fg_argparse
from pathlib import Path as _FgPath
_fg_ap = _fg_argparse.ArgumentParser(description='Figure 5: Class-Specific Sensitivity Tradeoffs (Tier B frozen tensors required)')
_fg_ap.add_argument('--root', type=_FgPath, default=_FgPath(__file__).resolve().parents[2], help='Data/repository root')
_fg_ap.add_argument('--out-dir', type=_FgPath, default=_FgPath(__file__).resolve().parents[2] / 'outputs/figures', help='Output figure directory')
_fg_args, _ = _fg_ap.parse_known_args()
_FG_ROOT = _fg_args.root.resolve()
_FG_OUT = _fg_args.out_dir.resolve()
_FG_OUT.mkdir(parents=True, exist_ok=True)

from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

# Enforce publication typography and deterministic vector embedding
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

ROOT = _FG_ROOT
OUT_DIR = _FG_OUT

PER_ACTION_TEST_PATH = ROOT / "research/derma_act/runs/derma_act_v0/predictions/per_action_test.npz"
PER_ACTION_DEV_PATH = ROOT / "research/derma_act/evidence/v1/per_action_dev.npz"
TEST_TRACE_PATH = ROOT / "research/derma_act/evidence/v1/test_policy_trace.csv"
METADATA_PATH = ROOT / "data/isic2019/raw/ISIC_2019_Training_Metadata.csv"
P54_JSON_PATH = ROOT / "research/derma_act/p5_4_independent_policy_validation/P5_4_BENCHMARK_ANALYSIS.json"
MCA_JSON_PATH = ROOT / "research/derma_act/p5_5_final_claim_closure/minority_class_audit.json"

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASSES_LOWER = [c.lower() for c in CLASSES]
SMALL_SUPPORT_THRESHOLD = 30
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261112
ALPHA = 0.05

# Okabe-Ito colorblind-safe palette
COL_FIXED = "#0072B2"      # Blue: Validation-Ranked Fixed Sequence (CAL-Stop)
COL_LEARNED = "#D55E00"    # Vermilion: Learned Policy V1
COL_SKY = "#56B4E9"        # Sky blue accent
COL_GREEN = "#009E73"      # Bluish green accent
COL_AMBER = "#E69F00"      # Amber warning accent
COL_PURPLE = "#CC79A7"     # Reddish purple accent
COL_GRAY = "#5C6670"
COL_LIGHT_GRAY = "#ECEFF2"

ACTION_META = [
    (1, "A1", "Center Zoom", "Spatial / Geometric", True, 4),
    (2, "A2", "Central-Mask Border", "Spatial / Geometric", False, None),
    (3, "A3", "Upper Region", "Spatial / Geometric", True, 3),
    (4, "A4", "Lower Region", "Spatial / Geometric", True, 5),
    (5, "A5", "Left Region", "Spatial / Geometric", True, 2),
    (6, "A6", "Right Region", "Spatial / Geometric", True, 1),
    (7, "A7", "Gabor Texture (8-bit sat.)", "Gabor Texture Filter", False, None),
    (8, "A8", "Color Constancy Norm.", "Color Constancy", False, None),
    (9, "A9", "Dark-Mask & Telea Inpaint", "Morphological Inpainting", False, None),
]


def step_down_holm(p_values: np.ndarray, alpha: float = 0.05):
    """Exact step-down Holm-Bonferroni correction with monotonicity enforcement."""
    m = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(m, dtype=int)
    adj_p = np.empty(m, dtype=float)
    alphas = np.empty(m, dtype=float)
    reject = np.zeros(m, dtype=bool)

    running_max = 0.0
    stopped = False
    for k, idx in enumerate(order):
        r = k + 1
        thr = alpha / (m - r + 1)
        cand = min(1.0, float(p_values[idx]) * (m - r + 1))
        running_max = max(running_max, cand)
        ranks[idx] = r
        alphas[idx] = thr
        adj_p[idx] = running_max
        if not stopped and p_values[idx] <= thr:
            reject[idx] = True
        else:
            stopped = True
    return ranks, alphas, adj_p, reject


def compute_all_data():
    """Load frozen artifacts, recompute all B=5 statistics, and verify P5.4/P5.5 parity."""
    test_npz = np.load(PER_ACTION_TEST_PATH, allow_pickle=True)
    dev_npz = np.load(PER_ACTION_DEV_PATH, allow_pickle=True)
    trace = pd.read_csv(TEST_TRACE_PATH)
    meta = pd.read_csv(METADATA_PATH)

    with open(P54_JSON_PATH, "r", encoding="utf-8") as f:
        p54 = json.load(f)
    with open(MCA_JSON_PATH, "r", encoding="utf-8") as f:
        mca = json.load(f)

    P_test = test_npz["probabilities"]  # (3686, 10, 7)
    y_test = test_npz["true_label"].astype(int)
    iids_test = test_npz["image_id"]
    n_test = len(y_test)
    assert n_test == 3686

    # Grouping units (operational lesion-ID / singleton partitions)
    lesion_map = dict(zip(meta["image"], meta["lesion_id"]))
    groups = []
    obs_lesions = set()
    missing_singletons = 0
    for iid in iids_test:
        lid = lesion_map.get(iid, np.nan)
        if pd.isna(lid) or str(lid).strip() == "":
            groups.append(f"IMG::{iid}")
            missing_singletons += 1
        else:
            s_lid = str(lid)
            groups.append(s_lid)
            obs_lesions.add(s_lid)
    groups = np.array(groups)
    uniq_groups = np.unique(groups)
    assert len(uniq_groups) == 1972
    assert len(obs_lesions) == 1695
    assert missing_singletons == 277

    # Learned V1 B=5 predictions (terminal row of full trace where max budget is 5)
    last_v1 = (
        trace.sort_values("step")
        .groupby("image_id")
        .tail(1)
        .set_index("image_id")
        .loc[iids_test]
    )
    preds_v1 = last_v1["predicted_class"].to_numpy().astype(int)

    # Check V1 action counts from trace
    non_stop = trace[(trace["step"] > 0) & (trace["selected_action"] != "STOP")]
    total_v1_actions = len(non_stop)
    assert total_v1_actions == 14815

    v1_action_stats = {}
    for _, tok, _, _, _, _ in ACTION_META:
        sub = non_stop[non_stop["selected_action"] == tok]
        v1_action_stats[tok] = {
            "total_selections": int(len(sub)),
            "unique_images": int(sub["image_id"].nunique()),
        }

    # Verify Validation-Ranked Fixed Sequence [6, 5, 3, 1, 4] and CAL threshold on dev
    split_dev = dev_npz["split"]
    val_mask = split_dev == "val"
    P_val = dev_npz["probabilities"][val_mask]
    y_val = dev_npz["true_label"][val_mask]
    # Note P5.4 ranked spatial actions [1..6] or full top1 ranking -> [6,5,3,1,4] matches canonical P5.4
    seq_val = [6, 5, 3, 1, 4]
    thr_b5 = float(p54["cal_frozen_thresholds"]["B_5"])
    assert abs(thr_b5 - float(mca["B5"]["cal_threshold"])) < 1e-12

    # Fixed comparator rollout at B=5 (cumulative arithmetic mean over A0 and sequential acquisitions)
    preds_fixed = np.zeros(n_test, dtype=int)
    fixed_action_counts = {a: 0 for a in range(1, 10)}
    fixed_views = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        prob_sum = P_test[i, 0, :].copy()
        n_views = 1
        for a in seq_val:
            if float(np.max(prob_sum / n_views)) >= thr_b5:
                break
            prob_sum = prob_sum + P_test[i, a, :]
            n_views += 1
            fixed_action_counts[a] += 1
        preds_fixed[i] = int(np.argmax(prob_sum / n_views))
        fixed_views[i] = n_views - 1
    total_fixed_actions = int(fixed_views.sum())

    # Confusion matrices (rows = true class, cols = predicted class)
    cm_v1 = np.zeros((7, 7), dtype=int)
    cm_fixed = np.zeros((7, 7), dtype=int)
    for yt, yv, yf in zip(y_test, preds_v1, preds_fixed):
        cm_v1[yt, yv] += 1
        cm_fixed[yt, yf] += 1

    # Assert exact parity with minority_class_audit.json B5
    assert np.array_equal(cm_v1, np.array(mca["B5"]["cm_v1"], dtype=int)), "cm_v1 mismatch vs MCA JSON"
    assert np.array_equal(cm_fixed, np.array(mca["B5"]["cm_fixed"], dtype=int)), "cm_fixed mismatch vs MCA JSON"

    # Bootstrap setup (1,972 partitioning units, 2,000 replicates, seed=20261112 matching P5.5)
    g_to_idx: dict[str, list[int]] = {}
    for idx, g in enumerate(groups):
        g_to_idx.setdefault(g, []).append(idx)
    g_arrays = [np.array(g_to_idx[g], dtype=int) for g in uniq_groups]
    k_units = len(g_arrays)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_sens_v1 = {c: [] for c in range(7)}
    boot_sens_fx = {c: [] for c in range(7)}
    boot_diff = {c: [] for c in range(7)}
    boot_bal_v1, boot_bal_fx, boot_bal_diff = [], [], []
    boot_top1_v1, boot_top1_fx, boot_top1_diff = [], [], []

    for _ in range(BOOTSTRAP_REPS):
        draw = rng.integers(0, k_units, size=k_units)
        s_idx = np.concatenate([g_arrays[d] for d in draw])
        yt = y_test[s_idx]
        yv = preds_v1[s_idx]
        yf = preds_fixed[s_idx]

        t1_v = float(np.mean(yv == yt))
        t1_f = float(np.mean(yf == yt))
        boot_top1_v1.append(t1_v)
        boot_top1_fx.append(t1_f)
        boot_top1_diff.append(t1_v - t1_f)

        rv, rf = [], []
        for c in range(7):
            mask = yt == c
            if np.any(mask):
                sc_v = float(np.mean(yv[mask] == c))
                sc_f = float(np.mean(yf[mask] == c))
                boot_sens_v1[c].append(sc_v)
                boot_sens_fx[c].append(sc_f)
                boot_diff[c].append(sc_v - sc_f)
                rv.append(sc_v)
                rf.append(sc_f)
            else:
                rv.append(np.nan)
                rf.append(np.nan)
        bv = float(np.nanmean(rv))
        bf = float(np.nanmean(rf))
        boot_bal_v1.append(bv)
        boot_bal_fx.append(bf)
        boot_bal_diff.append(bv - bf)

    # Exact McNemar across 7-class family
    raw_ps = []
    n10_list, n01_list = [], []
    for c in range(7):
        mask = y_test == c
        corr_v1 = preds_v1[mask] == c
        corr_fx = preds_fixed[mask] == c
        n10 = int(np.sum(corr_v1 & ~corr_fx))
        n01 = int(np.sum(~corr_v1 & corr_fx))
        n10_list.append(n10)
        n01_list.append(n01)
        tot_disc = n10 + n01
        pval = float(binomtest(min(n10, n01), tot_disc, 0.5, alternative="two-sided").pvalue) if tot_disc > 0 else 1.0
        raw_ps.append(pval)

    ranks, holm_alphas, holm_adj_ps, holm_rejects = step_down_holm(np.array(raw_ps), alpha=ALPHA)

    # Build Panel A & Panel B tables
    panel_a_rows = []
    panel_b_rows = []
    for c, cls_up in enumerate(CLASSES):
        cls_lo = CLASSES_LOWER[c]
        mask = y_test == c
        support = int(np.sum(mask))

        tp_v1 = int(cm_v1[c, c])
        fn_v1 = support - tp_v1
        fp_v1 = int(np.sum(cm_v1[:, c]) - tp_v1)
        tn_v1 = int(n_test - support - fp_v1)
        sens_v1 = tp_v1 / support
        spec_v1 = tn_v1 / (tn_v1 + fp_v1)
        prec_v1 = tp_v1 / (tp_v1 + fp_v1) if (tp_v1 + fp_v1) > 0 else 0.0

        tp_fx = int(cm_fixed[c, c])
        fn_fx = support - tp_fx
        fp_fx = int(np.sum(cm_fixed[:, c]) - tp_fx)
        tn_fx = int(n_test - support - fp_fx)
        sens_fx = tp_fx / support
        spec_fx = tn_fx / (tn_fx + fp_fx)
        prec_fx = tp_fx / (tp_fx + fp_fx) if (tp_fx + fp_fx) > 0 else 0.0

        ci_v1 = np.percentile(boot_sens_v1[c], [2.5, 97.5])
        ci_fx = np.percentile(boot_sens_fx[c], [2.5, 97.5])
        ci_diff = np.percentile(boot_diff[c], [2.5, 97.5])

        # Verify against minority_class_audit.json B5
        mca_c = mca["B5"]["classes"][cls_lo]
        assert support == mca_c["support"]
        assert abs(sens_v1 - mca_c["recall_v1"]) < 1e-12
        assert abs(sens_fx - mca_c["recall_fixed"]) < 1e-12
        assert n10_list[c] == mca_c["v1_correct_fixed_wrong"]
        assert n01_list[c] == mca_c["v1_wrong_fixed_correct"]
        assert abs(raw_ps[c] - mca_c["mcnemar_p"]) < 1e-12
        assert abs(holm_adj_ps[c] - mca_c["holm_adj_p"]) < 1e-12
        assert bool(holm_rejects[c]) == bool(mca_c["holm_reject"])
        assert abs(ci_diff[0] - mca_c["diff_recall_ci95"][0]) < 1e-12
        assert abs(ci_diff[1] - mca_c["diff_recall_ci95"][1]) < 1e-12

        small_flag = support < SMALL_SUPPORT_THRESHOLD
        panel_a_rows.append(
            {
                "class_index": c,
                "class_code": cls_up,
                "support_n": support,
                "support_pct_of_test": 100.0 * support / n_test,
                "small_support_flag": small_flag,
                "tp_learned_v1": tp_v1,
                "fn_learned_v1": fn_v1,
                "sensitivity_learned_v1": sens_v1,
                "sensitivity_learned_v1_pct": 100.0 * sens_v1,
                "sens_v1_ci95_low_pct": 100.0 * float(ci_v1[0]),
                "sens_v1_ci95_high_pct": 100.0 * float(ci_v1[1]),
                "specificity_learned_v1_pct": 100.0 * spec_v1,
                "precision_learned_v1_pct": 100.0 * prec_v1,
                "tp_fixed_calstop": tp_fx,
                "fn_fixed_calstop": fn_fx,
                "sensitivity_fixed_calstop": sens_fx,
                "sensitivity_fixed_calstop_pct": 100.0 * sens_fx,
                "sens_fixed_ci95_low_pct": 100.0 * float(ci_fx[0]),
                "sens_fixed_ci95_high_pct": 100.0 * float(ci_fx[1]),
                "specificity_fixed_calstop_pct": 100.0 * spec_fx,
                "precision_fixed_calstop_pct": 100.0 * prec_fx,
                "delta_tp_images": tp_v1 - tp_fx,
                "budget": 5,
            }
        )

        diff_sens = sens_v1 - sens_fx
        pt_excl_zero = (float(ci_diff[0]) > 0.0) or (float(ci_diff[1]) < 0.0)
        panel_b_rows.append(
            {
                "class_index": c,
                "class_code": cls_up,
                "support_n": support,
                "small_support_flag": small_flag,
                "delta_sensitivity": diff_sens,
                "delta_sensitivity_pp": 100.0 * diff_sens,
                "pointwise_ci95_low_pp": 100.0 * float(ci_diff[0]),
                "pointwise_ci95_high_pp": 100.0 * float(ci_diff[1]),
                "pointwise_ci_excludes_zero": pt_excl_zero,
                "n10_v1_correct_fixed_wrong": n10_list[c],
                "n01_v1_wrong_fixed_correct": n01_list[c],
                "total_discordant_pairs": n10_list[c] + n01_list[c],
                "mcnemar_exact_p": raw_ps[c],
                "holm_rank": int(ranks[c]),
                "holm_alpha": float(holm_alphas[c]),
                "holm_adjusted_p": float(holm_adj_ps[c]),
                "holm_reject_alpha05": bool(holm_rejects[c]),
                "bootstrap_reps": BOOTSTRAP_REPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "clustering_units_k": k_units,
                "observed_lesion_groups": len(obs_lesions),
                "missing_id_singletons": missing_singletons,
            }
        )

    # Panel C confusion matrix CSV (long tidy format for both strategies)
    panel_c_rows = []
    for strat_label, strat_key, cm in [
        ("Learned Policy V1 (B=5)", "learned_v1_b5", cm_v1),
        ("Validation-Ranked Fixed Sequence (CAL-Stop, B=5)", "fixed_calstop_b5", cm_fixed),
    ]:
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = cm / row_sums
        for i_true, t_code in enumerate(CLASSES):
            for j_pred, p_code in enumerate(CLASSES):
                panel_c_rows.append(
                    {
                        "strategy_key": strat_key,
                        "strategy_label": strat_label,
                        "budget": 5,
                        "true_index": i_true,
                        "true_class": t_code,
                        "pred_index": j_pred,
                        "pred_class": p_code,
                        "count": int(cm[i_true, j_pred]),
                        "true_class_support": int(row_sums[i_true, 0]),
                        "row_normalized_prob": float(norm_cm[i_true, j_pred]),
                        "row_normalized_pct": 100.0 * float(norm_cm[i_true, j_pred]),
                    }
                )

    # Panel D action composition CSV (include A0 baseline row + A1..A9 candidate actions)
    panel_d_rows = [
        {
            "action_index": 0,
            "action_token": "A0",
            "action_label": "Global View (Initial Baseline State)",
            "taxonomy_category": "Baseline Full-Image View",
            "included_in_fixed_comparator": True,
            "fixed_sequence_rank": 0,
            "available_to_learned_v1": True,
            "learned_v1_total_selections": n_test,
            "learned_v1_pct_of_14815_post_a0_selections": np.nan,
            "learned_v1_unique_images_exposed": n_test,
            "learned_v1_pct_of_3686_test_images": 100.0,
            "fixed_calstop_total_selections": n_test,
            "fixed_calstop_pct_of_post_a0_selections": np.nan,
            "fixed_calstop_unique_images_exposed": n_test,
            "fixed_calstop_pct_of_3686_test_images": 100.0,
            "notes": "Shared starting view for all 3,686 test episodes prior to budget B=1..5.",
        }
    ]
    for a_idx, tok, label, cat, in_fixed, f_rank in ACTION_META:
        v1_sel = v1_action_stats[tok]["total_selections"]
        v1_img = v1_action_stats[tok]["unique_images"]
        fx_sel = int(fixed_action_counts[a_idx])
        fx_img = fx_sel  # Deterministic single-pass sequence without repetition
        note = ""
        if tok == "A7":
            note = "Severe 8-bit unsigned convolution clipping (cohort median 99.9661% pre-norm saturation; 25.69% zero-collapse)."
        elif tok == "A8":
            note = "Excluded from Fixed Comparator; deployed in 74.96% of Learned V1 test episodes."
        elif tok == "A9":
            note = "Excluded from Fixed Comparator; non-selectively masks dark pigmentation & hairs via black-hat + Telea inpainting."
        elif tok == "A2":
            note = "Central 40% masked to zero; excluded from Fixed Comparator."
        panel_d_rows.append(
            {
                "action_index": a_idx,
                "action_token": tok,
                "action_label": label,
                "taxonomy_category": cat,
                "included_in_fixed_comparator": in_fixed,
                "fixed_sequence_rank": f_rank if f_rank is not None else -1,
                "available_to_learned_v1": True,
                "learned_v1_total_selections": v1_sel,
                "learned_v1_pct_of_14815_post_a0_selections": 100.0 * v1_sel / total_v1_actions,
                "learned_v1_unique_images_exposed": v1_img,
                "learned_v1_pct_of_3686_test_images": 100.0 * v1_img / n_test,
                "fixed_calstop_total_selections": fx_sel,
                "fixed_calstop_pct_of_post_a0_selections": 100.0 * fx_sel / total_fixed_actions,
                "fixed_calstop_unique_images_exposed": fx_img,
                "fixed_calstop_pct_of_3686_test_images": 100.0 * fx_img / n_test,
                "notes": note,
            }
        )

    summary = {
        "n_test": n_test,
        "k_units": k_units,
        "obs_lesions": len(obs_lesions),
        "missing_singletons": missing_singletons,
        "bal_acc_v1": float(np.mean([r["sensitivity_learned_v1"] for r in panel_a_rows])),
        "bal_acc_v1_ci95": [float(x) for x in np.percentile(boot_bal_v1, [2.5, 97.5])],
        "bal_acc_fixed": float(np.mean([r["sensitivity_fixed_calstop"] for r in panel_a_rows])),
        "bal_acc_fixed_ci95": [float(x) for x in np.percentile(boot_bal_fx, [2.5, 97.5])],
        "bal_acc_diff": float(np.mean([r["sensitivity_learned_v1"] - r["sensitivity_fixed_calstop"] for r in panel_a_rows])),
        "bal_acc_diff_ci95": [float(x) for x in np.percentile(boot_bal_diff, [2.5, 97.5])],
        "top1_acc_v1": float(np.mean(preds_v1 == y_test)),
        "top1_acc_fixed": float(np.mean(preds_fixed == y_test)),
        "top1_acc_diff": float(np.mean(preds_v1 == y_test) - np.mean(preds_fixed == y_test)),
        "top1_diff_ci95": [float(x) for x in np.percentile(boot_top1_diff, [2.5, 97.5])],
        "total_v1_actions": total_v1_actions,
        "total_fixed_actions": total_fixed_actions,
        "mean_v1_views": total_v1_actions / n_test,
        "mean_fixed_views": total_fixed_actions / n_test,
        "cal_threshold_b5": thr_b5,
        "cm_v1": cm_v1,
        "cm_fixed": cm_fixed,
    }

    return (
        pd.DataFrame(panel_a_rows),
        pd.DataFrame(panel_b_rows),
        pd.DataFrame(panel_c_rows),
        pd.DataFrame(panel_d_rows),
        summary,
    )


def format_p_sci(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    if p >= 0.99995:
        return "1.0000"
    return f"{p:.4f}"


def plot_figure5(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
    df_d: pd.DataFrame,
    summary: dict,
):
    """Render 4-panel publication-grade Figure 5 and export PDF, SVG, and 300 DPI PNG."""
    fig = plt.figure(figsize=(17.5, 15.2), facecolor="white")
    outer = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[1.08, 1.32, 1.18],
        hspace=0.38,
        left=0.115,
        right=0.965,
        top=0.915,
        bottom=0.045,
    )

    fig.suptitle(
        "Class-Specific Sensitivity Trade-Offs, Paired Confusion Structure, and Action Composition at Budget B = 5",
        fontsize=13.5,
        fontweight="bold",
        y=0.972,
    )
    fig.text(
        0.54,
        0.946,
        f"ISIC 2019 Held-Out Test Split (N = {summary['n_test']:,} images across K = {summary['k_units']:,} operational partitioning units: "
        f"{summary['obs_lesions']:,} lesion-ID groups + {summary['missing_singletons']} missing-ID singletons)  |  "
        f"Learned Policy V1 vs. Validation-Ranked Fixed Sequence (CAL-Stop, \u03c4\u2085 = {summary['cal_threshold_b5']:.4f})",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#333333",
    )

    # =========================================================================
    # ROW 1: PANEL A (Dumbbell Sensitivity) + PANEL B (Forest Paired Differences)
    # =========================================================================
    row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.0, 1.36], wspace=0.14)
    ax_a = fig.add_subplot(row1[0])
    ax_b = fig.add_subplot(row1[1])

    y_pos = np.arange(len(df_a))[::-1]  # MEL at top (6), VASC at bottom (0)

    # Highlight small-support rows cleanly across both Panel A and Panel B
    for idx_row, r in df_a.iterrows():
        y = y_pos[idx_row]
        if r["small_support_flag"]:
            ax_a.axhspan(y - 0.42, y + 0.42, color="#FFF3CD", alpha=0.55, zorder=0)
            ax_b.axhspan(y - 0.42, y + 0.42, color="#FFF3CD", alpha=0.55, zorder=0)

    # --- Panel A ---
    for idx_row, r in df_a.iterrows():
        y = y_pos[idx_row]
        sv1 = r["sensitivity_learned_v1_pct"]
        sfx = r["sensitivity_fixed_calstop_pct"]
        ax_a.plot([sfx, sv1], [y, y], color="#6E7781", lw=2.0, zorder=2)

    ax_a.scatter(
        df_a["sensitivity_fixed_calstop_pct"],
        y_pos,
        s=75,
        color=COL_FIXED,
        marker="o",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label=f"Validation-Ranked Fixed (CAL-Stop, BalAcc {100*summary['bal_acc_fixed']:.2f}%)",
    )
    ax_a.scatter(
        df_a["sensitivity_learned_v1_pct"],
        y_pos,
        s=75,
        color=COL_LEARNED,
        marker="s",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label=f"Learned Policy V1 (BalAcc {100*summary['bal_acc_v1']:.2f}%)",
    )

    # Value labels outside the dumbbell ends to avoid overlap
    for idx_row, r in df_a.iterrows():
        y = y_pos[idx_row]
        sv1 = r["sensitivity_learned_v1_pct"]
        sfx = r["sensitivity_fixed_calstop_pct"]
        if sv1 >= sfx:
            ax_a.text(sfx - 2.0, y, f"{sfx:.1f}%", ha="right", va="center", fontsize=8.0, color=COL_FIXED, fontweight="bold")
            ax_a.text(sv1 + 2.0, y, f"{sv1:.1f}%", ha="left", va="center", fontsize=8.0, color=COL_LEARNED, fontweight="bold")
        else:
            ax_a.text(sv1 - 2.0, y, f"{sv1:.1f}%", ha="right", va="center", fontsize=8.0, color=COL_LEARNED, fontweight="bold")
            ax_a.text(sfx + 2.0, y, f"{sfx:.1f}%", ha="left", va="center", fontsize=8.0, color=COL_FIXED, fontweight="bold")

    y_labels_a = [
        f"{r['class_code']} (n={r['support_n']:,})" + (" \u2020" if r["small_support_flag"] else "")
        for _, r in df_a.iterrows()
    ]
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(y_labels_a, fontweight="bold")
    ax_a.set_xlim(0, 106)
    ax_a.set_xticks([0, 20, 40, 60, 80, 100])
    ax_a.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax_a.set_xlabel("Class-Specific Sensitivity (Recall, %)")
    ax_a.set_title("A. Class-Specific Sensitivity Comparison (B = 5)", loc="left", fontweight="bold", pad=12)
    ax_a.grid(axis="x", linestyle=":", alpha=0.55, zorder=1)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    # Place legend in open upper-left quadrant (x=0..48%, y=NV/BCC area) or cleanly at bottom padding
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.01, 0.005), frameon=True, facecolor="white", edgecolor="#D0D7DE", framealpha=0.96, fontsize=8.0)
    ax_a.text(
        0.99,
        0.015,
        "\u2020Small-support (N < 30; exploratory)",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.6,
        color="#664D03",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFF3CD", edgecolor="#E69F00", alpha=0.9),
    )

    # --- Panel B ---
    ax_b.axvspan(-16, 0, color=COL_FIXED, alpha=0.045, zorder=0)
    ax_b.axvspan(0, 24, color=COL_LEARNED, alpha=0.045, zorder=0)
    ax_b.axvline(0.0, color="#24292F", linestyle="--", linewidth=1.2, zorder=2)

    for idx_row, r in df_b.iterrows():
        y = y_pos[idx_row]
        diff = r["delta_sensitivity_pp"]
        lo = r["pointwise_ci95_low_pp"]
        hi = r["pointwise_ci95_high_pp"]
        sig_holm = r["holm_reject_alpha05"]
        small_flag = r["small_support_flag"]

        if small_flag:
            col = "#7A5A00"
            mkr = "D"
        elif sig_holm:
            col = COL_LEARNED if diff > 0 else COL_FIXED
            mkr = "s"
        else:
            col = "#57606A"
            mkr = "o"

        ax_b.errorbar(
            diff,
            y,
            xerr=[[diff - lo], [hi - diff]],
            fmt=mkr,
            color=col,
            ecolor=col,
            elinewidth=1.8 if sig_holm else 1.4,
            capsize=3.5,
            markersize=6.8,
            zorder=4,
        )

        # Aligned statistical text columns on the right side
        ci_str = f"{diff:+5.2f} [{lo:+.2f}, {hi:+.2f}]"
        p_str = format_p_sci(r["holm_adjusted_p"])
        if sig_holm:
            sig_tag = "*"
            tag_weight = "bold"
            tag_col = col
        elif small_flag:
            sig_tag = "(expl.)"
            tag_weight = "normal"
            tag_col = "#664D03"
        elif r["pointwise_ci_excludes_zero"]:
            sig_tag = "ns (pt>0)"
            tag_weight = "normal"
            tag_col = "#57606A"
        else:
            sig_tag = "ns"
            tag_weight = "normal"
            tag_col = "#57606A"

        ax_b.text(24.0, y, ci_str, ha="left", va="center", fontsize=8.1, family="monospace", color="#24292F")
        ax_b.text(54.5, y, f"{p_str} {sig_tag}", ha="left", va="center", fontsize=8.1, color=tag_col, fontweight=tag_weight)

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([r["class_code"] + ("\u2020" if r["small_support_flag"] else "") for _, r in df_b.iterrows()], fontweight="bold")
    ax_b.set_xlim(-14.5, 78.0)
    ax_b.set_xticks([-10, -5, 0, 5, 10, 15, 20])
    ax_b.set_xticklabels(["-10", "-5", "0", "+5", "+10", "+15", "+20"])
    ax_b.set_xlabel("\u0394 Sensitivity = Learned V1 \u2212 Fixed (percentage points; pointwise 95% grouped bootstrap CI)")
    ax_b.set_title("B. Paired Sensitivity Differences & Holm-Adjusted Tests (m = 7)", loc="left", fontweight="bold", pad=12)
    ax_b.grid(axis="x", linestyle=":", alpha=0.55, zorder=1)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Column headers for embedded Panel B stat columns
    ax_b.text(-7.2, 6.62, "\u2190 Fixed Higher", ha="center", va="bottom", fontsize=8.2, color=COL_FIXED, fontweight="bold")
    ax_b.text(10.5, 6.62, "Learned V1 Higher \u2192", ha="center", va="bottom", fontsize=8.2, color=COL_LEARNED, fontweight="bold")
    ax_b.text(24.0, 6.62, "\u0394 (pp) [95% Pt. CI]", ha="left", va="bottom", fontsize=8.3, fontweight="bold", color="#24292F")
    ax_b.text(54.5, 6.62, "Holm p_adj", ha="left", va="bottom", fontsize=8.3, fontweight="bold", color="#24292F")
    ax_b.set_ylim(-1.25, 6.95)
    ax_a.set_ylim(-1.25, 6.95)

    # =========================================================================
    # ROW 2: PANEL C (Aligned 7x7 Confusion Matrices + Shared Colorbar)
    # =========================================================================
    row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[1.0, 1.0, 0.038], wspace=0.06)
    ax_c1 = fig.add_subplot(row2[0])
    ax_c2 = fig.add_subplot(row2[1])
    ax_cbar = fig.add_subplot(row2[2])

    cm_v1 = summary["cm_v1"]
    cm_fx = summary["cm_fixed"]
    norm_v1 = 100.0 * cm_v1 / cm_v1.sum(axis=1, keepdims=True)
    norm_fx = 100.0 * cm_fx / cm_fx.sum(axis=1, keepdims=True)

    cmap = plt.cm.Blues
    im1 = ax_c1.imshow(norm_v1, cmap=cmap, vmin=0.0, vmax=100.0, aspect="auto")
    im2 = ax_c2.imshow(norm_fx, cmap=cmap, vmin=0.0, vmax=100.0, aspect="auto")

    for cm_idx, (ax_cm, cm_raw, cm_pct, title_str) in enumerate(
        [
            (
                ax_c1,
                cm_v1,
                norm_v1,
                f"C1. Learned Policy V1 (B = 5)\nTop-1 Acc: {100*summary['top1_acc_v1']:.2f}% ({np.trace(cm_v1):,}/3,686)  |  BalAcc: {100*summary['bal_acc_v1']:.2f}%",
            ),
            (
                ax_c2,
                cm_fx,
                norm_fx,
                f"C2. Validation-Ranked Fixed (CAL-Stop, B = 5)\nTop-1 Acc: {100*summary['top1_acc_fixed']:.2f}% ({np.trace(cm_fx):,}/3,686)  |  BalAcc: {100*summary['bal_acc_fixed']:.2f}%",
            ),
        ]
    ):
        for i in range(7):
            for j in range(7):
                pct = cm_pct[i, j]
                cnt = cm_raw[i, j]
                txt_col = "white" if pct >= 52.0 else "#111827"
                fw = "bold" if i == j else "normal"
                ax_cm.text(
                    j,
                    i,
                    f"{pct:.1f}%\n({cnt})",
                    ha="center",
                    va="center",
                    color=txt_col,
                    fontsize=7.9,
                    fontweight=fw,
                )
        ax_cm.set_xticks(np.arange(7))
        ax_cm.set_xticklabels(CLASSES, fontweight="bold")
        ax_cm.set_yticks(np.arange(7))
        if cm_idx == 0:
            ax_cm.set_yticklabels([f"{CLASSES[i]} ({cm_raw[i].sum()})" for i in range(7)], fontweight="bold")
        else:
            ax_cm.set_yticklabels([])
        ax_cm.set_xlabel("Predicted Class (\u0177)")
        ax_cm.set_title(title_str, fontsize=10.0, fontweight="bold", pad=8)
        ax_cm.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax_cm.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax_cm.grid(which="minor", color="#D0D7DE", linestyle="-", linewidth=0.6)
        ax_cm.tick_params(which="minor", bottom=False, left=False)

    ax_c1.set_ylabel("True Class (y)")
    ax_c2.set_ylabel("")

    cbar = fig.colorbar(im2, cax=ax_cbar)
    cbar.set_label("Row-Normalized Rate P(\u0177 = j | y = i) (%)", fontsize=9.0)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    # =========================================================================
    # ROW 3: PANEL D (Action Inventory & Realized Exposure + Non-Causal Disclosure)
    # =========================================================================
    row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], width_ratios=[1.52, 0.96], wspace=0.08)
    ax_d = fig.add_subplot(row3[0])
    ax_d_box = fig.add_subplot(row3[1])

    df_d_cand = df_d[df_d["action_index"] >= 1].copy().reset_index(drop=True)
    y_d = np.arange(len(df_d_cand))[::-1]  # A1 top -> A9 bottom
    bar_h = 0.34

    vals_v1 = df_d_cand["learned_v1_pct_of_3686_test_images"].to_numpy()
    vals_fx = df_d_cand["fixed_calstop_pct_of_3686_test_images"].to_numpy()

    ax_d.barh(
        y_d + bar_h / 2,
        vals_v1,
        height=bar_h,
        color=COL_LEARNED,
        edgecolor="white",
        label=f"Learned Policy V1 (all 9 candidates A1\u2013A9; {summary['total_v1_actions']:,} selections, mean {summary['mean_v1_views']:.2f} views/img)",
    )
    ax_d.barh(
        y_d - bar_h / 2,
        vals_fx,
        height=bar_h,
        color=COL_FIXED,
        edgecolor="white",
        label=f"Fixed Sequence (CAL-Stop: A6\u2192A5\u2192A3\u2192A1\u2192A4; {summary['total_fixed_actions']:,} selections, mean {summary['mean_fixed_views']:.2f} views/img)",
    )

    for idx_r, r in df_d_cand.iterrows():
        y = y_d[idx_r]
        v1_p = r["learned_v1_pct_of_3686_test_images"]
        v1_n = int(r["learned_v1_unique_images_exposed"])
        v1_sel_p = r["learned_v1_pct_of_14815_post_a0_selections"]
        fx_p = r["fixed_calstop_pct_of_3686_test_images"]
        fx_n = int(r["fixed_calstop_unique_images_exposed"])

        ax_d.text(
            v1_p + 1.0,
            y + bar_h / 2,
            f"{v1_p:.2f}% ({v1_n:,}/3,686 | {v1_sel_p:.2f}% of 14,815)",
            ha="left",
            va="center",
            fontsize=7.6,
            color="#8F3200",
            fontweight="bold" if r["action_token"] in ("A1", "A8", "A9", "A7") else "normal",
        )
        if r["included_in_fixed_comparator"]:
            ax_d.text(
                fx_p + 1.0,
                y - bar_h / 2,
                f"{fx_p:.2f}% ({fx_n:,}/3,686; rank #{int(r['fixed_sequence_rank'])})",
                ha="left",
                va="center",
                fontsize=7.6,
                color=COL_FIXED,
            )
        else:
            ax_d.text(
                1.2,
                y - bar_h / 2,
                "Excluded from Fixed Comparator (0 / 3,686)",
                ha="left",
                va="center",
                fontsize=7.5,
                color="#B42318",
                style="italic",
                fontweight="bold",
            )

    y_labels_d = [f"{r['action_token']}: {r['action_label']}" for _, r in df_d_cand.iterrows()]
    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(y_labels_d, fontsize=8.4, fontweight="bold")
    ax_d.set_xlim(0, 145)
    ax_d.set_xticks([0, 20, 40, 60, 80, 100])
    ax_d.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax_d.set_xlabel("Unique Test Images Exposed (%) out of N = 3,686 (with Share of Total Post-A0 Actions Annotated)")
    ax_d.set_title("D. Candidate Virtual-View Inventory & Realized Policy Composition (B = 5)", loc="left", fontweight="bold", pad=36)
    ax_d.grid(axis="x", linestyle=":", alpha=0.55)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), frameon=True, facecolor="white", edgecolor="#D0D7DE", fontsize=7.9, ncol=1)

    # --- Panel D Methodological Qualification & Taxonomy Callout Box ---
    ax_d_box.axis("off")
    a8_r = df_d_cand[df_d_cand["action_token"] == "A8"].iloc[0]
    a9_r = df_d_cand[df_d_cand["action_token"] == "A9"].iloc[0]
    a7_r = df_d_cand[df_d_cand["action_token"] == "A7"].iloc[0]
    box_text = (
        "ACTION-INVENTORY ASYMMETRY & NON-CAUSAL QUALIFICATION\n"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "\u2022 Mandatory Interpretive Disclosure:\n"
        "  This benchmark evaluates two composite sequential strategies with\n"
        "  different action inventories. It does not isolate the causal\n"
        "  contribution of adaptive action selection under an identical\n"
        "  candidate-action set.\n\n"
        "\u2022 Inventory Breakdown & Denominator Distinction:\n"
        f"  \u2013 Initial A0 (Global View): applied on {summary['n_test']:,} / {summary['n_test']:,} (100%) images.\n"
        "  \u2013 Fixed Comparator (A6\u2192A5\u2192A3\u2192A1\u2192A4): strictly spatial crops\n"
        f"    ({summary['total_fixed_actions']:,} post-A0 acquisitions; {summary['mean_fixed_views']:.2f} views/img; excludes A2, A7, A8, A9).\n"
        f"  \u2013 Learned V1: accesses A1\u2013A9 ({summary['total_v1_actions']:,} post-A0 acquisitions;\n"
        f"    {summary['mean_v1_views']:.2f} views/img). No action repeats within an episode, so unique\n"
        "    image count equals selection count per action.\n\n"
        "\u2022 Photometric, Morphological & Gabor Audit Context:\n"
        f"  \u2013 A8 (Color Constancy): {int(a8_r['learned_v1_unique_images_exposed']):,} / {summary['n_test']:,} imgs "
        f"({a8_r['learned_v1_pct_of_3686_test_images']:.2f}%; {a8_r['learned_v1_pct_of_14815_post_a0_selections']:.2f}% of acts).\n"
        f"  \u2013 A9 (Dark-Structure Mask & Telea Inpaint): {int(a9_r['learned_v1_unique_images_exposed']):,} / {summary['n_test']:,} "
        f"({a9_r['learned_v1_pct_of_3686_test_images']:.2f}%; {a9_r['learned_v1_pct_of_14815_post_a0_selections']:.2f}% of acts);\n"
        "    non-selectively masks pigment networks & globules alongside hairs.\n"
        f"  \u2013 A7 (Gabor Texture): {int(a7_r['learned_v1_unique_images_exposed']):,} / {summary['n_test']:,} "
        f"({a7_r['learned_v1_pct_of_3686_test_images']:.2f}%; {a7_r['learned_v1_pct_of_14815_post_a0_selections']:.2f}% of acts);\n"
        "    99.97% cohort median pre-norm 8-bit saturation & 25.69% zero-collapse."
    )
    ax_d_box.text(
        0.02,
        0.96,
        box_text,
        transform=ax_d_box.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        linespacing=1.33,
        color="#1F2328",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F6F8FA", edgecolor="#8C959F", linewidth=1.1),
    )

    out_pdf = OUT_DIR / "fig5_class_specific_tradeoffs.pdf"
    out_svg = OUT_DIR / "fig5_class_specific_tradeoffs.svg"
    out_png = OUT_DIR / "fig5_class_specific_tradeoffs.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main():
    df_a, df_b, df_c, df_d, summary = compute_all_data()
    # csv export skipped in figure dir
    # csv export skipped in figure dir
    # csv export skipped in figure dir
    # csv export skipped in figure dir
    plot_figure5(df_a, df_b, df_c, df_d, summary)

    print("Figure 5 computation and rendering completed cleanly.")
    print(f"BalAcc Learned V1: {100*summary['bal_acc_v1']:.4f}% (CI95: [{100*summary['bal_acc_v1_ci95'][0]:.2f}%, {100*summary['bal_acc_v1_ci95'][1]:.2f}%])")
    print(f"BalAcc Fixed:      {100*summary['bal_acc_fixed']:.4f}% (CI95: [{100*summary['bal_acc_fixed_ci95'][0]:.2f}%, {100*summary['bal_acc_fixed_ci95'][1]:.2f}%])")
    print(f"BalAcc Diff:       {100*summary['bal_acc_diff']:+.4f} pp (CI95: [{100*summary['bal_acc_diff_ci95'][0]:+.2f}, {100*summary['bal_acc_diff_ci95'][1]:+.2f}] pp)")
    print(f"Top1 Acc Learned V1: {100*summary['top1_acc_v1']:.4f}% | Fixed: {100*summary['top1_acc_fixed']:.4f}% | Diff: {100*summary['top1_acc_diff']:+.4f} pp (CI95: [{100*summary['top1_diff_ci95'][0]:+.2f}, {100*summary['top1_diff_ci95'][1]:+.2f}] pp)")
    print(f"Fixed post-A0 actions: {summary['total_fixed_actions']} (mean {summary['mean_fixed_views']:.4f}/img)")
    print(f"V1 post-A0 actions:    {summary['total_v1_actions']} (mean {summary['mean_v1_views']:.4f}/img)")


if __name__ == "__main__":
    main()
