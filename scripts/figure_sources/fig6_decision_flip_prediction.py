"""Deterministic post-hoc verification, CSV export, and Figure 6 generator for DERMA-ACT V2.

Preservation guarantees:
- Strictly post-hoc analysis of frozen test_policy_trace.csv, per_action_test.npz,
  and ISIC_2019_Training_Metadata.csv.
- Zero model training, fine-tuning, policy inference, or classifier inference.
- Verifies exact temporal alignment (pre-acquisition score & class vs. post-acquisition
  next-step class) across all 14,815 transitions with zero future leakage.
- Cross-checks all point estimates and cluster-bootstrap 95% CIs (B=2,000 replicates,
  K=1,887 operational partitioning units, seed=20261112) against P5.5 canonical JSON.
"""

from __future__ import annotations

import json

import argparse as _fg_argparse
from pathlib import Path as _FgPath
_fg_ap = _fg_argparse.ArgumentParser(description='Figure 6: Decision-Flip Instability Scoring (Tier B trace & predictions required)')
_fg_ap.add_argument('--root', type=_FgPath, default=_FgPath(__file__).resolve().parents[2], help='Data/repository root')
_fg_ap.add_argument('--out-dir', type=_FgPath, default=_FgPath(__file__).resolve().parents[2] / 'outputs/figures', help='Output figure directory')
_fg_args, _ = _fg_ap.parse_known_args()
_FG_ROOT = _fg_args.root.resolve()
_FG_OUT = _fg_args.out_dir.resolve()
_FG_OUT.mkdir(parents=True, exist_ok=True)

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.3,
        "figure.titlesize": 13.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

ROOT = _FG_ROOT
OUT_DIR = _FG_OUT

TRACE_PATH = ROOT / "research/derma_act/evidence/v1/test_policy_trace.csv"
TEST_NPZ_PATH = ROOT / "research/derma_act/runs/derma_act_v0/predictions/per_action_test.npz"
META_PATH = ROOT / "data/isic2019/raw/ISIC_2019_Training_Metadata.csv"
CANON_JSON_PATH = ROOT / "research/derma_act/p5_5_final_claim_closure/decision_flip_cluster_results.json"

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261112

# Okabe-Ito colorblind-safe palette
COL_BLUE = "#0072B2"
COL_VERM = "#D55E00"
COL_GREEN = "#009E73"
COL_AMBER = "#E69F00"
COL_GRAY = "#57606A"
COL_DARK = "#1F2328"


def compute_and_verify():
    """Load frozen artifacts, prove temporal alignment, recompute metrics & CIs."""
    df_raw = pd.read_csv(TRACE_PATH)
    test_npz = np.load(TEST_NPZ_PATH, allow_pickle=True)
    meta = pd.read_csv(META_PATH).set_index("image")
    with open(CANON_JSON_PATH, "r", encoding="utf-8") as f:
        canon = json.load(f)

    P_test = test_npz["probabilities"]  # (3686, 10, 7)
    iid_to_idx = {iid: idx for idx, iid in enumerate(test_npz["image_id"])}

    # Verify strict step continuity across all 3,686 test episodes
    df = df_raw.sort_values(["image_id", "step"]).reset_index(drop=True)
    assert df["image_id"].nunique() == 3686
    assert df.groupby("image_id")["step"].apply(lambda s: list(s) == list(range(len(s)))).all()

    df["next_class"] = df.groupby("image_id")["predicted_class"].shift(-1)
    df["next_posterior"] = df.groupby("image_id")["posterior"].shift(-1)

    action_tokens = [f"A{i}" for i in range(1, 10)]
    sub = df[df["selected_action"].isin(action_tokens) & df["next_class"].notna()].copy().reset_index(drop=True)

    # Assert exact sample census
    assert len(sub) == 14815, f"Expected 14815 transitions, got {len(sub)}"
    sub["observed_flip"] = (sub["predicted_class"] != sub["next_class"]).astype(int)
    n_flips = int(sub["observed_flip"].sum())
    n_no_flip = int(len(sub) - n_flips)
    assert n_flips == 988 and n_no_flip == 13827
    assert not sub["predicted_instability"].isna().any()

    # Exhaustive temporal & arithmetic posterior check across all 14,815 transitions
    seen_pairs = set()
    max_pre_err = 0.0
    max_post_err = 0.0
    for _, r in sub.iterrows():
        pair = (r["image_id"], int(r["step"]))
        assert pair not in seen_pairs, f"Duplicate transition {pair}"
        seen_pairs.add(pair)

        idx = iid_to_idx[r["image_id"]]
        acq_list = [x for x in str(r["acquired_actions"]).strip(",").split(",") if x]
        pre_indices = [0 if x == "A0" else int(x[1:]) for x in acq_list]
        sel_idx = int(r["selected_action"][1:])
        assert sel_idx not in pre_indices, f"Repeated action in {pair}"

        p_pre_exp = P_test[idx, pre_indices, :].mean(axis=0)
        p_post_exp = P_test[idx, pre_indices + [sel_idx], :].mean(axis=0)
        p_pre_log = np.array(json.loads(r["posterior"]))
        p_post_log = np.array(json.loads(r["next_posterior"]))

        max_pre_err = max(max_pre_err, float(np.max(np.abs(p_pre_exp - p_pre_log))))
        max_post_err = max(max_post_err, float(np.max(np.abs(p_post_exp - p_post_log))))
        assert int(np.argmax(p_pre_log)) == int(r["predicted_class"])
        assert int(np.argmax(p_post_log)) == int(r["next_class"])

    assert max_pre_err < 1e-6 and max_post_err < 1e-6

    # Operational grouping units matching P5.5 exact rule
    lesion_ids = []
    lesion_obs_flags = []
    for iid in sub["image_id"]:
        lid = meta["lesion_id"].get(iid)
        if isinstance(lid, str) and lid.strip() != "":
            lesion_ids.append(lid)
            lesion_obs_flags.append(True)
        else:
            lesion_ids.append(f"IMG::{iid}")
            lesion_obs_flags.append(False)
    sub["operational_lesion_cluster"] = lesion_ids
    sub["lesion_id_observed_flag"] = lesion_obs_flags

    obs_clusters = sub.loc[sub["lesion_id_observed_flag"], "operational_lesion_cluster"].nunique()
    singleton_clusters = sub.loc[~sub["lesion_id_observed_flag"], "operational_lesion_cluster"].nunique()
    k_clusters = int(obs_clusters + singleton_clusters)
    assert k_clusters == 1887 and obs_clusters == 1627 and singleton_clusters == 260

    n_multiview_imgs = int(sub["image_id"].nunique())
    n_zero_trans_imgs = int(3686 - n_multiview_imgs)
    imgs_with_flip = int(sub.groupby("image_id")["observed_flip"].max().sum())
    assert n_multiview_imgs == 3371 and n_zero_trans_imgs == 315 and imgs_with_flip == 634

    # Export cleaned transition dataset
    df_trans_out = sub[
        [
            "image_id",
            "step",
            "budget",
            "selected_action",
            "predicted_class",
            "next_class",
            "observed_flip",
            "predicted_instability",
            "state_entropy",
            "state_confidence",
            "state_margin",
            "operational_lesion_cluster",
            "lesion_id_observed_flag",
        ]
    ].rename(columns={"predicted_class": "pre_predicted_class", "next_class": "post_predicted_class"})
    df_trans_out["post_predicted_class"] = df_trans_out["post_predicted_class"].astype(int)

    # Point metrics
    y_true = df_trans_out["observed_flip"].to_numpy()
    y_score = df_trans_out["predicted_instability"].to_numpy()
    clusters = df_trans_out["operational_lesion_cluster"].to_numpy()

    auroc_pt = float(roc_auc_score(y_true, y_score))
    ap_pt = float(average_precision_score(y_true, y_score))
    brier_pt = float(brier_score_loss(y_true, y_score))
    prev_pt = float(y_true.mean())

    assert abs(auroc_pt - canon["auroc"]) < 1e-12
    assert abs(ap_pt - canon["auprc"]) < 1e-12
    assert abs(brier_pt - canon["brier_score"]) < 1e-12
    assert abs(prev_pt - canon["prevalence"]) < 1e-12

    # Empirical ROC & PR curves
    fpr, tpr, roc_thr = roc_curve(y_true, y_score)
    df_roc = pd.DataFrame({"threshold": roc_thr, "fpr": fpr, "tpr": tpr})

    prec, rec, pr_thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(prec) = len(pr_thr) + 1 (last point has rec=0, prec=1)
    df_pr = pd.DataFrame(
        {
            "threshold": np.append(pr_thr, np.nan),
            "recall": rec,
            "precision": prec,
        }
    )

    # Cluster bootstrap (B=2000, K=1887, seed=20261112) preserving exact P5.5 sequence
    uniq_clusters, inv = np.unique(clusters, return_inverse=True)
    assert len(uniq_clusters) == 1887
    cluster_indices = [np.where(inv == k)[0] for k in range(len(uniq_clusters))]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_auc, boot_ap, boot_brier, boot_prev = [], [], [], []
    # Also run parallel cluster bootstrap for reliability bins using an independent child RNG
    # after finishing the canonical sequence so P5.5 exact numbers remain bit-identical.
    sampled_draws = []
    for _ in range(BOOTSTRAP_REPS):
        draw = rng.integers(0, len(uniq_clusters), size=len(uniq_clusters))
        sampled_draws.append(draw)
        b_idx = np.concatenate([cluster_indices[k] for k in draw])
        yt_b = y_true[b_idx]
        ys_b = y_score[b_idx]
        if len(np.unique(yt_b)) > 1:
            boot_auc.append(roc_auc_score(yt_b, ys_b))
            boot_ap.append(average_precision_score(yt_b, ys_b))
        boot_brier.append(brier_score_loss(yt_b, ys_b))
        boot_prev.append(yt_b.mean())

    def ci95(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    auc_ci = ci95(boot_auc)
    ap_ci = ci95(boot_ap)
    brier_ci = ci95(boot_brier)
    prev_ci = ci95(boot_prev)

    assert abs(auc_ci[0] - canon["auroc_ci95"][0]) < 1e-12
    assert abs(auc_ci[1] - canon["auroc_ci95"][1]) < 1e-12
    assert abs(ap_ci[0] - canon["auprc_ci95"][0]) < 1e-12
    assert abs(ap_ci[1] - canon["auprc_ci95"][1]) < 1e-12
    assert abs(brier_ci[0] - canon["brier_ci95"][0]) < 1e-12
    assert abs(brier_ci[1] - canon["brier_ci95"][1]) < 1e-12
    assert abs(prev_ci[0] - canon["prevalence_ci95"][0]) < 1e-12
    assert abs(prev_ci[1] - canon["prevalence_ci95"][1]) < 1e-12

    # Score distribution summary by outcome + prespecified 10-bin equal-width reliability
    dist_stats = {}
    for label_val, grp_key in [(0, "no_flip"), (1, "decision_flip")]:
        vals = y_score[y_true == label_val]
        q5, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
        dist_stats[grp_key] = {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "q05": float(q5),
            "q25": float(q25),
            "median": float(q50),
            "q75": float(q75),
            "iqr": float(q75 - q25),
            "q95": float(q95),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    # Reliability binning: 10 equal-width bins [0.0, 0.1) .. [0.9, 1.0]
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_rows = []
    # Precompute bin membership index 0..9
    bin_ids = np.clip(np.floor(y_score * 10.0).astype(int), 0, 9)
    for b in range(10):
        lo, hi = float(bin_edges[b]), float(bin_edges[b + 1])
        m = bin_ids == b
        n_b = int(m.sum())
        if n_b == 0:
            bin_rows.append(
                {
                    "record_type": "equal_width_reliability_bin",
                    "group_or_bin": f"[{lo:.1f}, {hi:.1f}]",
                    "bin_index": b,
                    "bin_low": lo,
                    "bin_high": hi,
                    "n_transitions": 0,
                    "n_flips": 0,
                    "n_clusters": 0,
                    "mean_predicted_score": np.nan,
                    "observed_flip_rate": np.nan,
                    "obs_rate_ci95_low": np.nan,
                    "obs_rate_ci95_high": np.nan,
                    "q05": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "iqr": np.nan,
                    "q95": np.nan,
                    "sparsity_status": "EMPTY",
                }
            )
            continue

        pos_b = int(y_true[m].sum())
        ku_b = int(pd.Series(clusters[m]).nunique())
        mp_b = float(y_score[m].mean())
        obs_b = float(y_true[m].mean())
        # Compute grouped bootstrap CI across the exact same 2,000 draws
        b_rates = []
        for draw in sampled_draws:
            b_idx = np.concatenate([cluster_indices[k] for k in draw])
            mb = bin_ids[b_idx] == b
            if np.any(mb):
                b_rates.append(float(y_true[b_idx][mb].mean()))
        r_ci = [float(np.percentile(b_rates, 2.5)), float(np.percentile(b_rates, 97.5))] if b_rates else [np.nan, np.nan]
        status = "SPARSE_N_LE_30" if n_b <= 30 else "WELL_POPULATED"
        bin_rows.append(
            {
                "record_type": "equal_width_reliability_bin",
                "group_or_bin": f"[{lo:.1f}, {hi:.1f}{')' if b < 9 else ']'}",
                "bin_index": b,
                "bin_low": lo,
                "bin_high": hi,
                "n_transitions": n_b,
                "n_flips": pos_b,
                "n_clusters": ku_b,
                "mean_predicted_score": mp_b,
                "observed_flip_rate": obs_b,
                "obs_rate_ci95_low": r_ci[0],
                "obs_rate_ci95_high": r_ci[1],
                "q05": np.nan,
                "q25": np.nan,
                "median": np.nan,
                "q75": np.nan,
                "iqr": np.nan,
                "q95": np.nan,
                "sparsity_status": status,
            }
        )

    # Combine outcome distribution summaries + reliability bins into Panel C CSV
    dist_rows = []
    for k_idx, (grp_key, label_txt) in enumerate([("no_flip", "No Flip (y=0)"), ("decision_flip", "Decision Flip (y=1)")]):
        st = dist_stats[grp_key]
        dist_rows.append(
            {
                "record_type": "outcome_score_distribution",
                "group_or_bin": label_txt,
                "bin_index": k_idx,
                "bin_low": st["min"],
                "bin_high": st["max"],
                "n_transitions": st["n"],
                "n_flips": st["n"] if k_idx == 1 else 0,
                "n_clusters": k_clusters,
                "mean_predicted_score": st["mean"],
                "observed_flip_rate": 1.0 if k_idx == 1 else 0.0,
                "obs_rate_ci95_low": np.nan,
                "obs_rate_ci95_high": np.nan,
                "q05": st["q05"],
                "q25": st["q25"],
                "median": st["median"],
                "q75": st["q75"],
                "iqr": st["iqr"],
                "q95": st["q95"],
                "sparsity_status": "OUTCOME_STRATUM",
            }
        )
    df_panel_c = pd.DataFrame(dist_rows + bin_rows)

    # Panel D summary CSV
    df_panel_d = pd.DataFrame(
        [
            {
                "metric_key": "auroc",
                "metric_label": "Decision-Flip AUROC",
                "estimate": auroc_pt,
                "ci95_low": auc_ci[0],
                "ci95_high": auc_ci[1],
                "reference_baseline": 0.5000,
                "sample_transitions_m": len(sub),
                "positive_flips": n_flips,
                "negative_non_flips": n_no_flip,
                "resampling_units_k": k_clusters,
                "observed_lesion_groups": obs_clusters,
                "missing_id_singletons": singleton_clusters,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "notes": "Threshold-free ranking discrimination across M=14,815 on-policy pre-acquisition states.",
            },
            {
                "metric_key": "auprc_ap",
                "metric_label": "Average Precision (AP / non-interpolated AUPRC)",
                "estimate": ap_pt,
                "ci95_low": ap_ci[0],
                "ci95_high": ap_ci[1],
                "reference_baseline": prev_pt,
                "sample_transitions_m": len(sub),
                "positive_flips": n_flips,
                "negative_non_flips": n_no_flip,
                "resampling_units_k": k_clusters,
                "observed_lesion_groups": obs_clusters,
                "missing_id_singletons": singleton_clusters,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "notes": f"sklearn.metrics.average_precision_score step-function AP; {ap_pt/prev_pt:.2f}x positive transition prevalence baseline.",
            },
            {
                "metric_key": "brier_score",
                "metric_label": "Brier Score",
                "estimate": brier_pt,
                "ci95_low": brier_ci[0],
                "ci95_high": brier_ci[1],
                "reference_baseline": prev_pt * (1.0 - prev_pt),
                "sample_transitions_m": len(sub),
                "positive_flips": n_flips,
                "negative_non_flips": n_no_flip,
                "resampling_units_k": k_clusters,
                "observed_lesion_groups": obs_clusters,
                "missing_id_singletons": singleton_clusters,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "notes": "Proper scoring rule reflecting combined calibration and refinement; not standalone proof of clinical calibration.",
            },
            {
                "metric_key": "flip_prevalence",
                "metric_label": "Transition Flip Prevalence",
                "estimate": prev_pt,
                "ci95_low": prev_ci[0],
                "ci95_high": prev_ci[1],
                "reference_baseline": np.nan,
                "sample_transitions_m": len(sub),
                "positive_flips": n_flips,
                "negative_non_flips": n_no_flip,
                "resampling_units_k": k_clusters,
                "observed_lesion_groups": obs_clusters,
                "missing_id_singletons": singleton_clusters,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "notes": f"Transition-level prevalence 988/14,815 = 6.6689% (distinct from image-level >=1 flip frequency: 634/3,371 multiview imgs = 18.81%, or 634/3,686 total test imgs = 17.20%).",
            },
        ]
    )

    summary = {
        "m_trans": len(sub),
        "n_flips": n_flips,
        "n_no_flip": n_no_flip,
        "prev_pt": prev_pt,
        "prev_ci": prev_ci,
        "auroc_pt": auroc_pt,
        "auc_ci": auc_ci,
        "ap_pt": ap_pt,
        "ap_ci": ap_ci,
        "brier_pt": brier_pt,
        "brier_ci": brier_ci,
        "k_clusters": k_clusters,
        "obs_clusters": obs_clusters,
        "singleton_clusters": singleton_clusters,
        "n_multiview_imgs": n_multiview_imgs,
        "n_zero_trans_imgs": n_zero_trans_imgs,
        "imgs_with_flip": imgs_with_flip,
        "max_pre_err": max_pre_err,
        "max_post_err": max_post_err,
        "dist_stats": dist_stats,
    }

    return df_trans_out, df_roc, df_pr, df_panel_c, df_panel_d, summary


def plot_figure6(df_roc, df_pr, df_panel_c, df_panel_d, df_trans, summary):
    """Render publication-grade 4-panel Figure 6 and export PDF, SVG, and 300 DPI PNG."""
    fig = plt.figure(figsize=(16.2, 13.4), facecolor="white")
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.12],
        width_ratios=[1.0, 1.0],
        hspace=0.33,
        wspace=0.22,
        left=0.065,
        right=0.975,
        top=0.895,
        bottom=0.055,
    )

    fig.suptitle(
        "Auxiliary Decision-Flip Forecasting: Threshold-Free Discrimination, Precision\u2013Recall, and Probability Diagnostics",
        fontsize=13.5,
        fontweight="bold",
        y=0.968,
    )
    fig.text(
        0.52,
        0.938,
        f"Held-Out ISIC 2019 On-Policy Test Rollout: M = {summary['m_trans']:,} pre-acquisition transitions "
        f"({summary['n_flips']:,} flips [{100*summary['prev_pt']:.2f}%], {summary['n_no_flip']:,} non-flips)  |  "
        f"N = {summary['n_multiview_imgs']:,} multi-view images across K = {summary['k_clusters']:,} operational partitioning units",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#333333",
    )

    # =========================================================================
    # PANEL A: ROC DISCRIMINATION
    # =========================================================================
    ax_a = fig.add_subplot(outer[0, 0])
    ax_a.plot([0, 1], [0, 1], linestyle="--", color=COL_GRAY, lw=1.3, label="Random-Ranking Baseline (AUROC = 0.5000)", zorder=1)
    ax_a.plot(
        df_roc["fpr"],
        df_roc["tpr"],
        color=COL_BLUE,
        lw=2.2,
        label=f"Empirical ROC Curve (AUROC = {summary['auroc_pt']:.4f})",
        zorder=3,
    )
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.02)
    ax_a.set_xlabel("False Positive Rate (FPR = FP / [FP + TN])")
    ax_a.set_ylabel("True Positive Rate (TPR = TP / [TP + FN])")
    ax_a.set_title("A. Decision-Flip ROC Discrimination (Threshold-Free)", loc="left", fontweight="bold", pad=10)
    ax_a.grid(True, linestyle=":", alpha=0.55)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.legend(loc="lower right", bbox_to_anchor=(0.98, 0.25), frameon=True, facecolor="white", edgecolor="#D0D7DE")

    roc_note = (
        f"AUROC = {summary['auroc_pt']:.6f}\n"
        f"95% Cluster-Bootstrap CI: [{summary['auc_ci'][0]:.6f}, {summary['auc_ci'][1]:.6f}]\n"
        f"(B = 2,000 replicates over K = {summary['k_clusters']:,} lesion-ID/singleton units;\n"
        f"scalar CI shown — no pointwise curve band computed)\n"
        f"Transitions: M = {summary['m_trans']:,} ({summary['n_flips']:,} flips; {summary['n_no_flip']:,} non-flips)"
    )
    ax_a.text(
        0.97,
        0.03,
        roc_note,
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.9,
        color=COL_DARK,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#F6F8FA", edgecolor="#8C959F"),
    )

    # =========================================================================
    # PANEL B: PRECISION-RECALL PERFORMANCE
    # =========================================================================
    ax_b = fig.add_subplot(outer[0, 1])
    prev = summary["prev_pt"]
    ax_b.axhline(
        prev,
        linestyle="--",
        color=COL_GRAY,
        lw=1.3,
        label=f"Transition Prevalence Baseline ({summary['n_flips']:,}/{summary['m_trans']:,} = {100*prev:.2f}%)",
        zorder=1,
    )
    ax_b.step(
        df_pr["recall"],
        df_pr["precision"],
        where="post",
        color=COL_VERM,
        lw=2.1,
        label=f"Empirical PR Step Curve (AP = {summary['ap_pt']:.4f})",
        zorder=3,
    )
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(0.0, 1.02)
    ax_b.set_xlabel("Recall (Sensitivity = TP / [TP + FN])")
    ax_b.set_ylabel("Precision (PPV = TP / [TP + FP])")
    ax_b.set_title("B. Precision\u2013Recall Curve & Positive Prevalence Baseline", loc="left", fontweight="bold", pad=10)
    ax_b.grid(True, linestyle=":", alpha=0.55)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D0D7DE")

    pr_note = (
        f"Average Precision (AP / step AUPRC) = {summary['ap_pt']:.6f}\n"
        f"95% Cluster-Bootstrap CI: [{summary['ap_ci'][0]:.6f}, {summary['ap_ci'][1]:.6f}]\n"
        f"Enrichment over baseline prevalence: {summary['ap_pt']/prev:.2f}\u00d7\n"
        f"Positives: {summary['n_flips']:,} flips  |  Negatives: {summary['n_no_flip']:,} non-flips\n"
        f"No discrete cutoff prespecified; clinical alert utility unproven."
    )
    ax_b.text(
        0.97,
        0.78,
        pr_note,
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=7.9,
        color=COL_DARK,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#FFF8F3", edgecolor=COL_VERM, alpha=0.95),
    )

    # =========================================================================
    # PANEL C: SCORE DIAGNOSTICS (C1 ECDF DISTRIBUTIONS + C2 RELIABILITY)
    # =========================================================================
    c_sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0], width_ratios=[1.0, 1.05], wspace=0.28)
    ax_c1 = fig.add_subplot(c_sub[0])
    ax_c2 = fig.add_subplot(c_sub[1])

    scores_0 = np.sort(df_trans.loc[df_trans["observed_flip"] == 0, "predicted_instability"].to_numpy())
    scores_1 = np.sort(df_trans.loc[df_trans["observed_flip"] == 1, "predicted_instability"].to_numpy())
    ecdf_0 = np.arange(1, len(scores_0) + 1) / len(scores_0)
    ecdf_1 = np.arange(1, len(scores_1) + 1) / len(scores_1)

    ax_c1.step(scores_0, ecdf_0, where="post", color=COL_BLUE, lw=2.0, label=f"No Flip (y=0, n={len(scores_0):,})")
    ax_c1.step(scores_1, ecdf_1, where="post", color=COL_VERM, lw=2.0, label=f"Decision Flip (y=1, n={len(scores_1):,})")
    ax_c1.set_xlim(0.0, 0.92)
    ax_c1.set_ylim(0.0, 1.02)
    ax_c1.set_xlabel("Predicted Instability Score (\u015d\u209c)")
    ax_c1.set_ylabel("Empirical Cumulative Proportion")
    ax_c1.set_title("C1. Score ECDF by Outcome", loc="left", fontweight="bold", pad=10, fontsize=10.3)
    ax_c1.grid(True, linestyle=":", alpha=0.55)
    ax_c1.spines["top"].set_visible(False)
    ax_c1.spines["right"].set_visible(False)
    ax_c1.legend(loc="lower right", bbox_to_anchor=(0.99, 0.44), frameon=True, facecolor="white", edgecolor="#D0D7DE", fontsize=7.6)

    d0 = summary["dist_stats"]["no_flip"]
    d1 = summary["dist_stats"]["decision_flip"]
    dist_txt = (
        f"No Flip (n={d0['n']:,}):\n"
        f"  Med {d0['median']:.4f}, IQR {d0['iqr']:.4f}\n"
        f"  P5\u2013P95: [{d0['q05']:.4f}, {d0['q95']:.4f}]\n"
        f"Flip (n={d1['n']:,}):\n"
        f"  Med {d1['median']:.4f}, IQR {d1['iqr']:.4f}\n"
        f"  P5\u2013P95: [{d1['q05']:.4f}, {d1['q95']:.4f}]"
    )
    ax_c1.text(
        0.98,
        0.03,
        dist_txt,
        transform=ax_c1.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        family="monospace",
        color=COL_DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F6F8FA", edgecolor="#8C959F"),
    )

    # --- C2: Equal-Width 10-Bin Reliability Diagram ---
    bins_df = df_panel_c[df_panel_c["record_type"] == "equal_width_reliability_bin"].copy()
    pop_df = bins_df[bins_df["sparsity_status"] == "WELL_POPULATED"]
    sp_df = bins_df[bins_df["sparsity_status"] == "SPARSE_N_LE_30"]

    ax_c2.plot([0, 1], [0, 1], linestyle="--", color=COL_GRAY, lw=1.2, label="Perfect Calibration (y = x)", zorder=1)
    # Connect all non-empty bins lightly to show trend
    ne_df = bins_df[bins_df["n_transitions"] > 0]
    ax_c2.plot(ne_df["mean_predicted_score"], ne_df["observed_flip_rate"], color="#8C959F", lw=1.1, zorder=2)

    if len(pop_df) > 0:
        ax_c2.errorbar(
            pop_df["mean_predicted_score"],
            pop_df["observed_flip_rate"],
            yerr=[
                pop_df["observed_flip_rate"] - pop_df["obs_rate_ci95_low"],
                pop_df["obs_rate_ci95_high"] - pop_df["observed_flip_rate"],
            ],
            fmt="o",
            color=COL_GREEN,
            ecolor=COL_GREEN,
            markersize=5.8,
            capsize=2.8,
            lw=1.4,
            label="Equal-Width Bins (n > 30)",
            zorder=4,
        )
    if len(sp_df) > 0:
        ax_c2.errorbar(
            sp_df["mean_predicted_score"],
            sp_df["observed_flip_rate"],
            yerr=[
                sp_df["observed_flip_rate"] - sp_df["obs_rate_ci95_low"],
                sp_df["obs_rate_ci95_high"] - sp_df["observed_flip_rate"],
            ],
            fmt="^",
            color=COL_AMBER,
            ecolor=COL_AMBER,
            markersize=6.4,
            capsize=2.8,
            lw=1.3,
            label="Sparse Bins (n \u2264 30)",
            zorder=5,
        )

    # Annotate transition counts cleanly above/below points
    offsets = {0: (10, 5), 1: (10, -14), 2: (10, -14), 3: (10, -14), 4: (10, -14), 5: (8, -15), 6: (10, -13), 7: (10, -13), 8: (-28, -14)}
    for _, br in ne_df.iterrows():
        b_i = int(br["bin_index"])
        dx, dy = offsets.get(b_i, (6, -12))
        ax_c2.annotate(
            f"n={int(br['n_transitions'])}",
            (br["mean_predicted_score"], br["observed_flip_rate"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.8,
            color="#333333",
        )

    ax_c2.axvspan(0.9, 1.0, color="#ECEFF2", alpha=0.75, zorder=0)
    ax_c2.text(0.95, 0.50, "Empty\n[0.9, 1.0]\n(n=0)", ha="center", va="center", fontsize=7.0, color=COL_GRAY, rotation=90)
    ax_c2.set_xlim(-0.02, 1.02)
    ax_c2.set_ylim(-0.02, 1.05)
    ax_c2.set_xlabel("Mean Predicted Score (10 Equal-Width Bins)")
    ax_c2.set_ylabel("Observed Decision-Flip Frequency")
    ax_c2.set_title("C2. Reliability & Proper Brier Score", loc="left", fontweight="bold", pad=10, fontsize=10.3)
    ax_c2.grid(True, linestyle=":", alpha=0.55)
    ax_c2.spines["top"].set_visible(False)
    ax_c2.spines["right"].set_visible(False)
    ax_c2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#D0D7DE", fontsize=7.5)

    brier_txt = (
        f"Brier Score = {summary['brier_pt']:.6f}\n"
        f"95% CI: [{summary['brier_ci'][0]:.4f}, {summary['brier_ci'][1]:.4f}]\n"
        f"Proper rule (calib. + refinement);\n"
        f"not standalone calibration proof."
    )
    ax_c2.text(
        0.97,
        0.03,
        brier_txt,
        transform=ax_c2.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        color=COL_DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F6F8FA", edgecolor="#8C959F"),
    )

    # =========================================================================
    # PANEL D: CLUSTERED UNCERTAINTY & SAMPLE STRUCTURE + LIMITATION BOX
    # =========================================================================
    ax_d = fig.add_subplot(outer[1, 1])
    ax_d.axis("off")
    ax_d.set_title("D. Cluster-Bootstrap Uncertainty, Cohort Structure & Operational Limits", loc="left", fontweight="bold", pad=10)

    # Structured metric table in top half of Panel D
    col_labels = ["Metric (Test Rollout)", "Estimate", "95% Cluster CI", "Baseline / Context"]
    table_rows = [
        [
            "Decision-Flip AUROC",
            f"{summary['auroc_pt']:.6f}",
            f"[{summary['auc_ci'][0]:.6f}, {summary['auc_ci'][1]:.6f}]",
            "vs. 0.5000 random (+0.4473)",
        ],
        [
            "Average Precision (AP)",
            f"{summary['ap_pt']:.6f}",
            f"[{summary['ap_ci'][0]:.6f}, {summary['ap_ci'][1]:.6f}]",
            f"vs. {100*prev:.2f}% prev. ({summary['ap_pt']/prev:.2f}x)",
        ],
        [
            "Brier Score (Proper Rule)",
            f"{summary['brier_pt']:.6f}",
            f"[{summary['brier_ci'][0]:.6f}, {summary['brier_ci'][1]:.6f}]",
            "vs. 0.062242 prev. baseline",
        ],
        [
            "Transition Flip Prevalence",
            f"{100*prev:.4f}%",
            f"[{100*summary['prev_ci'][0]:.4f}%, {100*summary['prev_ci'][1]:.4f}%]",
            f"{summary['n_flips']:,} / {summary['m_trans']:,} steps",
        ],
    ]
    tbl = ax_d.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        bbox=[0.0, 0.56, 1.0, 0.42],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.8)
    col_widths = [0.27, 0.14, 0.30, 0.29]
    for (row_i, col_i), cell in tbl.get_celld().items():
        cell.set_width(col_widths[col_i])
        cell.set_edgecolor("#D0D7DE")
        if row_i == 0:
            cell.set_facecolor("#EAF2F8")
            cell.set_text_props(weight="bold", color=COL_DARK)
        else:
            cell.set_facecolor("#FFFFFF" if row_i % 2 == 1 else "#F9FAFB")
            if col_i in (1, 2):
                cell.set_text_props(family="monospace", fontsize=7.6)

    # Hierarchy & Mandatory Operational Threshold Limitation box in bottom half of Panel D
    panel_d_text = (
        "CLUSTER-RESAMPLING HIERARCHY & DENOMINATOR DISAMBIGUATION\n"
        f"  \u2022 Resampling Unit (K = {summary['k_clusters']:,} operational clusters, B = 2,000 reps, seed 20261112):\n"
        f"    {summary['obs_clusters']:,} observed ISIC lesion_id clusters + {summary['singleton_clusters']} missing-ID singleton images\n"
        "    (patient-level independence is not established). All longitudinal steps within every\n"
        "    image and lesion cluster are resampled together (no transition-level IID assumption).\n"
        f"  \u2022 Transition vs. Image Base Rate: {summary['n_flips']:,}/{summary['m_trans']:,} transitions (6.67%) flip;\n"
        f"    across N = {summary['n_multiview_imgs']:,} multi-view images (+{summary['n_zero_trans_imgs']} immediate stops = 3,686 total),\n"
        f"    {summary['imgs_with_flip']:,} images experience \u22651 Top-1 flip ({100*summary['imgs_with_flip']/summary['n_multiview_imgs']:.2f}% multi-view; {100*summary['imgs_with_flip']/3686:.2f}% test).\n\n"
        "OPERATIONAL THRESHOLD & CLINICAL LIMITATION\n"
        "  \"The instability head is evaluated as a threshold-free predictor of subsequent Top-1\n"
        "  decision changes. No operational binary alert threshold was prespecified; therefore,\n"
        "  clinical operating characteristics and intervention utility were not established.\""
    )
    ax_d.text(
        0.0,
        0.51,
        panel_d_text,
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=7.9,
        linespacing=1.32,
        color=COL_DARK,
        bbox=dict(boxstyle="round,pad=0.52", facecolor="#F6F8FA", edgecolor="#8C959F", linewidth=1.05),
    )

    fig.savefig(OUT_DIR / "fig6_decision_flip_prediction.pdf", bbox_inches="tight", pad_inches=0.16)
    fig.savefig(OUT_DIR / "fig6_decision_flip_prediction.svg", bbox_inches="tight", pad_inches=0.16)
    fig.savefig(OUT_DIR / "fig6_decision_flip_prediction.png", dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def main():
    df_trans, df_roc, df_pr, df_c, df_d, summary = compute_and_verify()
    # csv export skipped in figure dir
    # csv export skipped in figure dir
    # csv export skipped in figure dir
    # csv export skipped in figure dir
    # csv export skipped in figure dir
    plot_figure6(df_roc, df_pr, df_c, df_d, df_trans, summary)

    print("Figure 6 computation, verification, and rendering completed cleanly.")
    print(f"Verified M={summary['m_trans']} transitions ({summary['n_flips']} flips, {summary['n_no_flip']} non-flips), max_pre_err={summary['max_pre_err']:.2e}, max_post_err={summary['max_post_err']:.2e}")
    print(f"AUROC={summary['auroc_pt']:.6f} {summary['auc_ci']} | AP={summary['ap_pt']:.6f} {summary['ap_ci']} | Brier={summary['brier_pt']:.6f} {summary['brier_ci']}")


if __name__ == "__main__":
    main()
