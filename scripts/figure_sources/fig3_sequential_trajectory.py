"""
Figure 3: Stored Sequential Inference Trajectory & Decision Transition Visualization (Patch 1.1).
DERMA-ACT Trajectory Trace for ISIC_0000003 (MEL -> NV Decision Flip under Budget Exhaustion).
Strict Non-Bayesian Arithmetic Fusion & Descriptive Trajectory Terminology.

Panels:
  - Panel A: Sequential Action Timeline (A0 -> A5 -> A1 -> A4 -> A3 -> A8)
  - Panel B: Seven-Class Probability Evolution (Seven-Class Running Fused Probabilities)
  - Panel C: Arithmetic Fusion Mechanism at Decision Flip (Step t=2)
  - Panel D: Decision-State Diagnostics Along One Illustrative Stored Trajectory

Generates:
  - fig3_sequential_trajectory_v1_1.svg
  - fig3_sequential_trajectory_v1_1.pdf
  - fig3_sequential_trajectory_v1_1.png (300 DPI)
"""
from pathlib import Path

import argparse as _fg_argparse
from pathlib import Path as _FgPath
_fg_ap = _fg_argparse.ArgumentParser(description='Figure 3: Sequential Trajectory Case Study (Tier B: requires rendered ISIC crop thumbnails)')
_fg_ap.add_argument('--root', type=_FgPath, default=_FgPath(__file__).resolve().parents[2], help='Data/repository root')
_fg_ap.add_argument('--out-dir', type=_FgPath, default=_FgPath(__file__).resolve().parents[2] / 'outputs/figures', help='Output figure directory')
_fg_args, _ = _fg_ap.parse_known_args()
_FG_ROOT = _fg_args.root.resolve()
_FG_OUT = _fg_args.out_dir.resolve()
_FG_OUT.mkdir(parents=True, exist_ok=True)

import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_figure():
    # Set high-quality sans-serif typography
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    fig = plt.figure(figsize=(16, 13), dpi=300)

    # Master Color Palette (Okabe-Ito & Publication Tones)
    c_mel = "#D55E00"      # Melanoma (Vermillion / Red-Orange)
    c_nv = "#0072B2"       # Melanocytic Nevi (Blue)
    c_bcc = "#E69F00"      # Basal Cell Carcinoma (Amber)
    c_akiec = "#CC79A7"    # Actinic Keratosis (Reddish Purple)
    c_bkl = "#56B4E9"      # Benign Keratosis (Sky Blue)
    c_df = "#009E73"       # Dermatofibroma (Bluish Green)
    c_vasc = "#718096"     # Vascular Lesions (Slate Gray)
    c_darkgray = "#2D3748" # Dark text
    c_gray = "#A0AEC0"     # Light grid / subtle text

    # Paths
    v1_1_dir = _FG_ROOT / "data/figure3_trajectory"
    if not (v1_1_dir / "FIGURE_3_TRAJECTORY_DATA.csv").exists():
        raise FileNotFoundError("Missing Tier B Figure 3 trajectory data/views under --root")
    v1_1_dir.mkdir(parents=True, exist_ok=True)
    data_csv = v1_1_dir / "FIGURE_3_TRAJECTORY_DATA.csv"
    df = pd.read_csv(data_csv)
    views_dir = v1_1_dir / "views"

    # Overall Figure Background & Headers
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_xlim(0, 100)
    ax_bg.set_ylim(0, 100)
    ax_bg.axis("off")

    # Overall Figure Title (Updated per Section 2: Stored Sequential Inference Trajectory)
    ax_bg.text(50, 98.4, "DERMA-ACT: Stored Sequential Inference Trajectory & Decision Transition Visualization",
               ha="center", va="center", fontsize=13.5, fontweight="bold", color="#1A202C")
    ax_bg.text(50, 96.9, "Illustrative Stored Test Trajectory (ISIC_0000003) | Dataset Reference Label: NV (Melanocytic Nevus) | Final Termination: Budget Exhausted (B=5)",
               ha="center", va="center", fontsize=8.8, color="#4A5568")

    # =========================================================================
    # PANEL A: SEQUENTIAL ACTION TIMELINE
    # =========================================================================
    panel_a = patches.Rectangle((2, 69.5), 96, 25.5, facecolor="#F8FAFC", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax_bg.add_patch(panel_a)
    ax_bg.text(3.5, 93.4, "(A) SEQUENTIAL ACTION TIMELINE & TOP-1 PREDICTION TRANSITIONS", fontsize=10.2, fontweight="bold", color="#1A202C")
    ax_bg.text(3.5, 92.0, "Ordered Inspection Steps (A0 -> A5 -> A1 -> A4 -> A3 -> A8) | Stored Active Trajectory from Test Partition", fontsize=7.6, color="#718096")

    # 6 Timeline Cards (t = 0 to 5)
    card_xs = [3.5, 19.5, 35.5, 51.5, 67.5, 83.5]
    card_w = 13.0
    card_h = 20.0
    card_y = 70.8

    action_tokens = ["A0", "A5", "A1", "A4", "A3", "A8"]
    action_titles = [
        "A0: Global View",
        "A5: Left Region",
        "A1: Center Zoom",
        "A4: Lower Region",
        "A3: Upper Region",
        "A8: Color Suppressed"
    ]
    view_files = [
        "A0_global_view.png",
        "A5_left_region.png",
        "A1_center_zoom.png",
        "A4_lower_region.png",
        "A3_upper_region.png",
        "A8_color_suppressed.png"
    ]

    for t in range(6):
        r = df.iloc[t]
        cx = card_xs[t]
        tok = action_tokens[t]
        top1 = r["top1_class"]
        conf = r["top1_confidence"] * 100.0
        marg = r["top2_margin"] * 100.0
        
        # Color based on predicted class: Red for MEL, Blue for NV
        border_col = c_mel if top1 == "MEL" else c_nv
        is_flip = (t == 2)
        is_terminal = (t == 5)

        # Draw card box
        card_box = patches.FancyBboxPatch(
            (cx, card_y), card_w, card_h,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor="#FFFFFF", edgecolor=border_col, linewidth=1.8 if (is_flip or is_terminal) else 1.2, zorder=2
        )
        ax_bg.add_patch(card_box)

        # Card Header Badge
        header_badge = patches.FancyBboxPatch(
            (cx + 0.4, card_y + card_h - 2.0), card_w - 0.8, 1.6,
            boxstyle="round,pad=0,rounding_size=0.4",
            facecolor="#2D3748", edgecolor="none", zorder=3
        )
        ax_bg.add_patch(header_badge)
        ax_bg.text(cx + card_w/2, card_y + card_h - 1.2, f"Step t={t}: {tok}",
                   ha="center", va="center", fontsize=6.8, fontweight="bold", color="#FFFFFF", zorder=4)

        # View Thumbnail
        thumb_ax = fig.add_axes([(cx + (card_w - 8.2)/2) / 100.0, (card_y + 8.8) / 100.0, 8.2 / 100.0, 8.2 / 100.0])
        img = cv2.imread(str(views_dir / view_files[t]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        thumb_ax.imshow(img_rgb)
        thumb_ax.set_xticks([])
        thumb_ax.set_yticks([])
        for spine in thumb_ax.spines.values():
            spine.set_edgecolor("#CBD5E0")
            spine.set_linewidth(0.8)

        # Action Name below thumbnail
        ax_bg.text(cx + card_w/2, card_y + 8.0, action_titles[t],
                   ha="center", va="center", fontsize=6.0, fontweight="bold", color="#2D3748", zorder=3)

        # Prediction Status Box
        pred_box = patches.FancyBboxPatch(
            (cx + 0.6, card_y + 3.2), card_w - 1.2, 4.2,
            boxstyle="round,pad=0,rounding_size=0.4",
            facecolor="#F7FAFC", edgecolor=border_col, linewidth=1.0, zorder=3
        )
        ax_bg.add_patch(pred_box)
        
        status_label = "INITIAL" if t == 0 else ("DECISION FLIP" if is_flip else ("FINAL (B=5)" if is_terminal else "INTERMEDIATE"))
        ax_bg.text(cx + card_w/2, card_y + 6.6, status_label,
                   ha="center", va="center", fontsize=5.0, fontweight="bold", color=border_col, zorder=4)
        ax_bg.text(cx + card_w/2, card_y + 5.2, f"Top-1: {top1} ({conf:.1f}%)",
                   ha="center", va="center", fontsize=6.2, fontweight="bold", color=border_col, zorder=4)
        ax_bg.text(cx + card_w/2, card_y + 4.0, f"Top-2 Margin: {marg:.1f}%",
                   ha="center", va="center", fontsize=5.4, color="#4A5568", zorder=4)

        # Bottom detail: next action / stopping
        if t < 5:
            next_act = r["selected_next_action"]
            ax_bg.text(cx + card_w/2, card_y + 1.8, f"Next: {next_act}",
                       ha="center", va="center", fontsize=5.4, fontweight="bold", color="#2B6CB0", zorder=3)
            ax_bg.text(cx + card_w/2, card_y + 0.8, f"Budget left: {5 - t}",
                       ha="center", va="center", fontsize=4.8, color="#718096", zorder=3)
        else:
            ax_bg.text(cx + card_w/2, card_y + 1.8, "STOP: Budget Exhausted",
                       ha="center", va="center", fontsize=5.2, fontweight="bold", color="#C53030", zorder=3)
            ax_bg.text(cx + card_w/2, card_y + 0.8, "Max budget reached (B=5)",
                       ha="center", va="center", fontsize=4.8, color="#718096", zorder=3)

        # Connector arrow between cards
        if t < 5:
            arr_x1 = cx + card_w + 0.3
            arr_x2 = card_xs[t+1] - 0.3
            arr_y = card_y + card_h / 2
            ax_bg.annotate("", xy=(arr_x2, arr_y), xytext=(arr_x1, arr_y),
                           arrowprops=dict(arrowstyle="-|>", color="#718096", lw=1.6, mutation_scale=12), zorder=5)

    # =========================================================================
    # PANEL B: SEVEN-CLASS RUNNING FUSED PROBABILITIES
    # =========================================================================
    # Updated Header per Section 2: Seven-Class Running Fused Probabilities
    panel_b = patches.Rectangle((2, 35.5), 47.0, 32.5, facecolor="#F8FAFC", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax_bg.add_patch(panel_b)
    ax_bg.text(3.5, 66.4, "(B) SEVEN-CLASS RUNNING FUSED PROBABILITIES ACROSS STATES", fontsize=10.2, fontweight="bold", color="#1A202C")
    ax_bg.text(3.5, 65.0, "Seven-Class Running Fused Probabilities | Crossover at Step t=2 (MEL -> NV)", fontsize=7.6, color="#718096")

    # Inset Axes for Panel B Line Chart
    ax_b = fig.add_axes([0.055, 0.380, 0.420, 0.250])
    steps = np.arange(6)

    # Extract all 7 class probabilities
    p_mel = df["fused_prob_MEL"].values
    p_nv = df["fused_prob_NV"].values
    p_bcc = df["fused_prob_BCC"].values
    p_akiec = df["fused_prob_AKIEC"].values
    p_bkl = df["fused_prob_BKL"].values
    p_df = df["fused_prob_DF"].values
    p_vasc = df["fused_prob_VASC"].values

    # Highlight crossover / decision flip region
    ax_b.axvspan(1.5, 2.5, color="#FEFCBF", alpha=0.45, zorder=1, label="Decision Flip Window (t=2)")
    ax_b.axvline(2.0, color="#D69E2E", linestyle="--", linewidth=1.2, zorder=2)

    # Plot lines for all 7 classes
    ax_b.plot(steps, p_mel, color=c_mel, marker="o", linewidth=2.4, markersize=6.5, label="MEL (Melanoma)", zorder=5)
    ax_b.plot(steps, p_nv, color=c_nv, marker="s", linewidth=2.4, markersize=6.5, label="NV (Melanocytic Nevi)", zorder=5)
    ax_b.plot(steps, p_bcc, color=c_bcc, marker="^", linewidth=1.2, markersize=4.5, linestyle=":", label="BCC", zorder=3)
    ax_b.plot(steps, p_akiec, color=c_akiec, marker="v", linewidth=1.2, markersize=4.5, linestyle=":", label="AKIEC", zorder=3)
    ax_b.plot(steps, p_bkl, color=c_bkl, marker="d", linewidth=1.2, markersize=4.5, linestyle="-.", label="BKL", zorder=3)
    ax_b.plot(steps, p_df, color=c_df, marker="p", linewidth=1.2, markersize=4.5, linestyle=":", label="DF", zorder=3)
    ax_b.plot(steps, p_vasc, color=c_vasc, marker="x", linewidth=1.2, markersize=4.5, linestyle=":", label="VASC", zorder=3)

    # Annotate Decision Flip Point at t=2 (Placed below line in clear zone)
    ax_b.annotate("Decision Flip\nMEL (40.3%) < NV (50.9%)",
                  xy=(2.0, 0.45), xytext=(1.0, 0.20),
                  arrowprops=dict(arrowstyle="->", color="#1A202C", lw=1.2),
                  fontsize=6.8, fontweight="bold", color="#1A202C",
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFBEB", edgecolor="#D69E2E", lw=1.0),
                  zorder=10)

    # Annotate Final State at t=5
    ax_b.text(5.08, p_nv[-1], f"NV: {p_nv[-1]*100:.1f}%\n(Reference Match)", fontsize=6.2, fontweight="bold", color=c_nv, va="center")
    ax_b.text(5.08, p_mel[-1], f"MEL: {p_mel[-1]*100:.1f}%", fontsize=6.2, fontweight="bold", color=c_mel, va="center")

    ax_b.set_xlim(-0.2, 5.8)
    ax_b.set_ylim(0.0, 0.72)
    ax_b.set_xticks(steps)
    ax_b.set_xticklabels([f"t={t}\n({tok})" for t, tok in enumerate(action_tokens)], fontsize=6.8)
    ax_b.set_xlabel("Sequential Trajectory Step (Action Acquired)", fontsize=7.4, fontweight="bold", color="#2D3748")
    ax_b.set_ylabel("Fused Class Probability $p_t(c)$", fontsize=7.4, fontweight="bold", color="#2D3748")
    ax_b.grid(True, linestyle="--", alpha=0.5, color="#CBD5E0")
    ax_b.tick_params(axis="both", labelsize=6.8)

    # Legend for Panel B
    leg_b = ax_b.legend(loc="upper left", fontsize=5.8, framealpha=0.92, edgecolor="#CBD5E0", ncol=2)
    leg_b.set_zorder(8)

    # =========================================================================
    # PANEL C: RECURSIVE ARITHMETIC PREDICTION FUSION MECHANISM
    # =========================================================================
    panel_c = patches.Rectangle((51, 35.5), 47.0, 32.5, facecolor="#F8FAFC", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax_bg.add_patch(panel_c)
    ax_bg.text(52.5, 66.4, "(C) ARITHMETIC FUSION MECHANISM AT DECISION FLIP (STEP t=2)", fontsize=10.2, fontweight="bold", color="#1A202C")
    ax_bg.text(52.5, 65.0, "p2 = (2*p1 + p(A1)) / 3 | Exact Recursive Arithmetic Running Mean (Not Bayesian)", fontsize=7.6, color="#718096")

    # Inset Axes for Panel C Grouped Bar Chart
    ax_c = fig.add_axes([0.545, 0.380, 0.420, 0.250])

    classes_disp = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    x_indices = np.arange(len(classes_disp))
    bar_width = 0.26

    # Extract probability vectors at t=1, new A1, and t=2
    p1_vec = [df.iloc[1][f"fused_prob_{c}"] for c in classes_disp]
    p_a1_vec = [df.iloc[2][f"new_view_prob_{c}"] for c in classes_disp]
    p2_vec = [df.iloc[2][f"fused_prob_{c}"] for c in classes_disp]

    # Bar Labels updated per Section 2:
    # "Previous Fused Probability Vector", "New Virtual-View Probability Vector", "Updated Fused Probability Vector"
    bars1 = ax_c.bar(x_indices - bar_width, p1_vec, width=bar_width, color="#CBD5E0", edgecolor="#718096", linewidth=0.8,
                     label="Previous Fused Vector $p_1$ ({A0, A5}, $n_1=2$)")
    bars2 = ax_c.bar(x_indices, p_a1_vec, width=bar_width, color="#ECC94B", edgecolor="#B7791F", linewidth=0.8,
                     label="New Virtual-View Vector $p(A_1)$ (Zoom, $n=1$)")
    bars3 = ax_c.bar(x_indices + bar_width, p2_vec, width=bar_width, color="#3182CE", edgecolor="#2B6CB0", linewidth=0.8,
                     label="Updated Fused Vector $p_2$ ({A0,A5,A1}, $n_2=3$)")

    # Value labels on top of MEL and NV bars
    ax_c.text(x_indices[0] - bar_width, p1_vec[0] + 0.02, f"{p1_vec[0]*100:.1f}%", ha="center", fontsize=5.4, color="#2D3748")
    ax_c.text(x_indices[0], p_a1_vec[0] + 0.02, f"{p_a1_vec[0]*100:.1f}%", ha="center", fontsize=5.4, color="#B7791F", fontweight="bold")
    ax_c.text(x_indices[0] + bar_width, p2_vec[0] + 0.02, f"{p2_vec[0]*100:.1f}%", ha="center", fontsize=5.4, color="#C53030", fontweight="bold")

    ax_c.text(x_indices[1] - bar_width, p1_vec[1] + 0.02, f"{p1_vec[1]*100:.1f}%", ha="center", fontsize=5.4, color="#2D3748")
    ax_c.text(x_indices[1], p_a1_vec[1] + 0.02, f"{p_a1_vec[1]*100:.1f}%", ha="center", fontsize=5.4, color="#B7791F", fontweight="bold")
    ax_c.text(x_indices[1] + bar_width, p2_vec[1] + 0.02, f"{p2_vec[1]*100:.1f}%", ha="center", fontsize=5.4, color="#2B6CB0", fontweight="bold")

    # Formula Explanatory Box inside Panel C (Strict Non-Bayesian Arithmetic Average)
    formula_text = (
        r"$\mathbf{Recursive\ Arithmetic\ Prediction\ Fusion:}$" + "\n"
        r"$p_{t+1} = \frac{n_t \cdot p_t + p(a_{t+1})}{n_t + 1}, \quad n_{t+1} = n_t + 1$" + "\n"
        r"$\mathrm{At\ Step\ 2:}\ p_2 = \frac{2}{3} p_1 + \frac{1}{3} p(A_1)$" + "\n"
        r"$\mathrm{NV:}\ \frac{2}{3}(42.0\%) + \frac{1}{3}(68.7\%) = \mathbf{50.9\%}$" + "\n"
        r"$\mathrm{MEL:}\ \frac{2}{3}(48.7\%) + \frac{1}{3}(23.4\%) = \mathbf{40.3\%}$"
    )
    ax_c.text(0.98, 0.96, formula_text, transform=ax_c.transAxes, fontsize=5.7,
              verticalalignment="top", horizontalalignment="right",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#CBD5E0", lw=1.0))

    ax_c.set_ylim(0.0, 0.78)
    ax_c.set_xticks(x_indices)
    ax_c.set_xticklabels(classes_disp, fontsize=6.8, fontweight="bold")
    ax_c.set_xlabel("Dermatological Lesion Diagnostic Class", fontsize=7.4, fontweight="bold", color="#2D3748")
    ax_c.set_ylabel("Probability Value", fontsize=7.4, fontweight="bold", color="#2D3748")
    ax_c.grid(True, linestyle="--", alpha=0.5, color="#CBD5E0", axis="y")
    ax_c.tick_params(axis="both", labelsize=6.8)

    leg_c = ax_c.legend(loc="upper left", fontsize=5.6, framealpha=0.92, edgecolor="#CBD5E0")
    leg_c.set_zorder(10)

    # =========================================================================
    # PANEL D: DECISION-STATE DIAGNOSTICS ALONG ONE ILLUSTRATIVE STORED TRAJECTORY
    # =========================================================================
    # Updated Subtitle per Section 3: Decision-State Diagnostics Along One Illustrative Stored Trajectory
    panel_d = patches.Rectangle((2, 2.5), 96, 31.0, facecolor="#F8FAFC", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax_bg.add_patch(panel_d)
    ax_bg.text(3.5, 32.0, "(D) DECISION-STATE DIAGNOSTICS ALONG ONE ILLUSTRATIVE STORED TRAJECTORY", fontsize=10.2, fontweight="bold", color="#1A202C")
    ax_bg.text(3.5, 30.6, "Descriptive properties of this single trajectory; not a general causal or population-level claim", fontsize=7.6, color="#718096")

    # Subplot D1: Top-1 Confidence
    ax_d1 = fig.add_axes([0.055, 0.065, 0.200, 0.215])
    confs = df["top1_confidence"].values * 100.0
    ax_d1.plot(steps, confs, color="#2B6CB0", marker="o", linewidth=2.0, markersize=5.5)
    ax_d1.axvspan(1.5, 2.5, color="#FEFCBF", alpha=0.45)
    ax_d1.set_title("Top-1 Confidence (%)", fontsize=7.6, fontweight="bold", color="#1A202C")
    ax_d1.set_xticks(steps)
    ax_d1.set_xticklabels([f"t={t}" for t in steps], fontsize=6.5)
    ax_d1.set_ylabel("Confidence (%)", fontsize=6.8, color="#2D3748")
    ax_d1.grid(True, linestyle="--", alpha=0.5, color="#CBD5E0")
    ax_d1.tick_params(axis="both", labelsize=6.5)
    for idx, (x, y) in enumerate(zip(steps, confs)):
        y_offset = -2.2 if idx == 1 else 1.2
        ax_d1.text(x, y + y_offset, f"{y:.1f}%", ha="center", fontsize=5.2, color="#2B6CB0", fontweight="bold")
    ax_d1.set_ylim(43.0, 66.0)

    # Subplot D2: Top-2 Margin
    ax_d2 = fig.add_axes([0.295, 0.065, 0.200, 0.215])
    margins = df["top2_margin"].values * 100.0
    ax_d2.plot(steps, margins, color="#D69E2E", marker="s", linewidth=2.0, markersize=5.5)
    ax_d2.axvspan(1.5, 2.5, color="#FEFCBF", alpha=0.45)
    ax_d2.set_title("Top-2 Margin: p(1) - p(2) (%)", fontsize=7.6, fontweight="bold", color="#1A202C")
    ax_d2.set_xticks(steps)
    ax_d2.set_xticklabels([f"t={t}" for t in steps], fontsize=6.5)
    ax_d2.set_ylabel("Margin (%)", fontsize=6.8, color="#2D3748")
    ax_d2.grid(True, linestyle="--", alpha=0.5, color="#CBD5E0")
    ax_d2.tick_params(axis="both", labelsize=6.5)
    for idx, (x, y) in enumerate(zip(steps, margins)):
        y_offset = 2.2 if idx == 1 else 1.4
        ax_d2.text(x, y + y_offset, f"{y:.1f}%", ha="center", fontsize=5.2, color="#B7791F", fontweight="bold")
    ax_d2.set_ylim(0.0, 36.0)

    # Subplot D3: Predictive Shannon Entropy (Updated per Section 2: Predictive Shannon Entropy)
    ax_d3 = fig.add_axes([0.535, 0.065, 0.200, 0.215])
    entropies = df["shannon_entropy"].values
    ax_d3.plot(steps, entropies, color="#805AD5", marker="^", linewidth=2.0, markersize=5.5)
    ax_d3.axvspan(1.5, 2.5, color="#FEFCBF", alpha=0.45)
    ax_d3.set_title("Predictive Shannon Entropy (nats)", fontsize=7.6, fontweight="bold", color="#1A202C")
    ax_d3.set_xticks(steps)
    ax_d3.set_xticklabels([f"t={t}" for t in steps], fontsize=6.5)
    ax_d3.set_ylabel("Predictive Entropy $H(p_t)$", fontsize=6.8, color="#2D3748")
    ax_d3.grid(True, linestyle="--", alpha=0.5, color="#CBD5E0")
    ax_d3.tick_params(axis="both", labelsize=6.5)
    for idx, (x, y) in enumerate(zip(steps, entropies)):
        y_offset = 0.015 if idx != 3 else -0.022
        ax_d3.text(x, y + y_offset, f"{y:.3f}", ha="center", fontsize=5.2, color="#6B46C1", fontweight="bold")
    ax_d3.set_ylim(0.91, 1.15)

    # Subplot D4: Budget Remaining & Terminal Callout
    ax_d4 = fig.add_axes([0.775, 0.065, 0.200, 0.215])
    budgets = df["budget_remaining"].values
    ax_d4.step(steps, budgets, color="#E53E3E", where="mid", linewidth=2.0, marker="o", markersize=5.0)
    ax_d4.set_title("Remaining View Budget ($B - t$)", fontsize=7.6, fontweight="bold", color="#1A202C")
    ax_d4.set_xticks(steps)
    ax_d4.set_xticklabels([f"t={t}" for t in steps], fontsize=6.5)
    ax_d4.set_ylabel("Remaining Budget", fontsize=6.8, color="#2D3748")
    ax_d4.grid(True, linestyle="--", alpha=0.5, color="#CBD5E0")
    ax_d4.tick_params(axis="both", labelsize=6.5)
    for x, y in zip(steps, budgets):
        ax_d4.text(x, y + 0.28, f"B={y}", ha="center", fontsize=5.2, color="#C53030", fontweight="bold")
    ax_d4.text(5.0, 0.9, "Terminal:\nBudget=0", ha="center", fontsize=5.2, color="#C53030", fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF5F5", edgecolor="#E53E3E", lw=0.8))
    ax_d4.set_ylim(-0.5, 6.0)

    # Save deliverables to V1_1 directory
    svg_path = _FG_OUT / "fig3_sequential_trajectory.svg"
    pdf_path = _FG_OUT / "fig3_sequential_trajectory.pdf"
    png_path = _FG_OUT / "fig3_sequential_trajectory.png"

    plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Successfully generated Figure 3 V1.1 deliverables:")
    print(f"  - SVG: {svg_path} ({svg_path.stat().st_size} bytes)")
    print(f"  - PDF: {pdf_path} ({pdf_path.stat().st_size} bytes)")
    print(f"  - PNG: {png_path} ({png_path.stat().st_size} bytes)")


if __name__ == "__main__":
    create_figure()
