"""
Figure 1 Architecture Source V3.2 — Final Conformal Order-Statistic & Grouping Alignment (P7.1/P8.0).
Online Runtime Perception Loop (Panel A) vs. Group-Aware Offline Evaluation & Comparative Benchmarking (Panel B).

Generates:
  - fig1_architecture_v3_2.svg (Editable Vector SVG)
  - fig1_architecture_v3_2.pdf (Publication Vector PDF)
  - fig1_architecture_v3_2.png (300 DPI high-resolution raster)
"""
from pathlib import Path

import argparse as _fg_argparse
from pathlib import Path as _FgPath
_fg_ap = _fg_argparse.ArgumentParser(description='Figure 1: DERMA-ACT System Architecture (Tier A standalone)')
_fg_ap.add_argument('--root', type=_FgPath, default=_FgPath(__file__).resolve().parents[2], help='Data/repository root')
_fg_ap.add_argument('--out-dir', type=_FgPath, default=_FgPath(__file__).resolve().parents[2] / 'outputs/figures', help='Output figure directory')
_fg_args, _ = _fg_ap.parse_known_args()
_FG_ROOT = _fg_args.root.resolve()
_FG_OUT = _fg_args.out_dir.resolve()
_FG_OUT.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_figure():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Okabe-Ito Color Palette & Publication Tones
    c_blue = "#0072B2"      # Classifiers, Backbone, & Output
    c_orange = "#D55E00"    # Policy, Utility, Actions & Recurrence
    c_green = "#009E73"     # Conformal & Calibration
    c_amber = "#E69F00"     # Auxiliary Instability Heads & Decision Flips
    c_skyblue = "#56B4E9"   # Virtual Views & Robustness Battery
    c_darkgray = "#2D3748"  # High-contrast primary text & borders
    c_gray = "#718096"      # Subtitles & secondary notes
    c_red = "#C53030"       # Negative Benchmark / Fixed Comparator
    c_purple = "#7B2CBF"    # Offline Data Distribution Bus

    def draw_card(x, y, w, h, title, subtitle="", lines=None, bg="#FFFFFF", border="#CBD5E0", 
                  title_color="#1A202C", title_size=8.5, sub_size=7.2, body_size=6.5, lw=1.2, radius=1.0):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=bg, edgecolor=border, linewidth=lw, zorder=2
        )
        ax.add_patch(box)

        # Header positioning
        cur_y = y + h - 1.4
        if title:
            ax.text(x + w/2, cur_y, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
            cur_y -= 1.2
        
        if subtitle:
            ax.text(x + w/2, cur_y, subtitle, ha="center", va="center", fontsize=sub_size, fontweight="bold", color=c_gray, zorder=3)
            cur_y -= 1.3
        
        # Body text with consistent vertical line spacing
        if lines:
            line_spacing = (cur_y - y - 0.7) / max(len(lines), 1)
            line_spacing = min(line_spacing, 1.3)  # cap spacing
            for line, weight, color in lines:
                fweight = "bold" if weight == "bold" else "normal"
                fstyle = "italic" if weight == "italic" else "normal"
                ax.text(x + w/2, cur_y, line, ha="center", va="center", fontsize=body_size, 
                        fontweight=fweight, fontstyle=fstyle, color=color, zorder=3)
                cur_y -= line_spacing

    def draw_arrow(start, end, style="-", color="#2D3748", lw=1.3, rad=0.0):
        connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                linewidth=lw,
                linestyle=style,
                shrinkA=2,
                shrinkB=2,
                connectionstyle=connectionstyle,
                mutation_scale=10
            ),
            zorder=4
        )

    # =========================================================================
    # OVERALL HEADER & PANEL BOUNDING RECTANGLES
    # =========================================================================
    ax.text(50, 98.3, "DERMA-ACT: System Architecture & Group-Aware Offline Evaluation Framework",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#1A202C")
    ax.text(50, 96.8, "Per-Image Sequential Perception Loop (Panel A) vs. Group-Aware Offline Comparative Benchmarking (Panel B)",
            ha="center", va="center", fontsize=9.0, color=c_gray)

    # Panel A Bounding Box
    panel_a = patches.Rectangle((2, 51.5), 96, 43.5, facecolor="#F8FAFC", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax.add_patch(panel_a)
    ax.text(3.5, 93.5, "(A) ONLINE RUNTIME PERCEPTION & SEQUENTIAL FUSION LOOP", fontsize=10.5, fontweight="bold", color="#1A202C")
    ax.text(3.5, 92.0, "Per-image Sequential Inference Workflow | Solid Arrows: Synchronous Feed-Forward Execution | Dotted Orange Arrow: Recurrent State Update", fontsize=7.8, color=c_gray)

    # Panel B Bounding Box
    panel_b = patches.Rectangle((2, 3), 96, 46.5, facecolor="#F8FAFC", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax.add_patch(panel_b)
    ax.text(3.5, 47.8, "(B) GROUP-AWARE OFFLINE EVALUATION AND COMPARATIVE BENCHMARKING", fontsize=10.5, fontweight="bold", color="#1A202C")
    ax.text(3.5, 46.3, "Four Distinct Offline Evaluation Analyses | Dashed Lines: Offline Dataset Provenance & Routing (Analyses Share Partitions; Not Statistically Independent)", fontsize=7.8, color=c_gray)

    # =========================================================================
    # PANEL A: RUNTIME PERCEPTION LOOP (CLOCKWISE FLOW)
    # =========================================================================
    # 1. Input Image
    draw_card(3.5, 75.5, 14.5, 14.5, "1. Input Dermoscopic Image", "RGB Acquisition", [
        ("Raw photograph:", "bold", c_darkgray),
        ("x in R^(H x W x 3)", "normal", c_darkgray),
        ("Bicubic resize to", "normal", c_darkgray),
        ("224 x 224 x 3", "bold", c_darkgray),
        ("ImageNet norm.", "normal", c_gray)
    ], bg="#FFFFFF", border=c_darkgray)

    # 2. Baseline Global View A0
    draw_card(20.0, 75.5, 15.0, 14.5, "2. Global View A0", "Baseline Classifier", [
        ("Frozen Backbone:", "bold", c_blue),
        ("EfficientNet-B4 (17.9M)", "normal", c_darkgray),
        ("Embedding: z_0 in R^1792", "normal", c_darkgray),
        ("Softmax: p_0 in Delta^6", "bold", c_blue),
        ("Initial count: n_0 = 1", "bold", c_darkgray)
    ], bg="#EBF8FF", border=c_blue)

    # 3. State Vector Assembly (Decomposition sums exactly to 28)
    draw_card(37.0, 75.5, 16.5, 14.5, "3. State Vector s_t", "s_t in R^28 (Exact)", [
        ("Fused Prob: p_t in Delta^6 (d=7)", "bold", c_darkgray),
        ("Shannon Entropy: H(p_t) (d=1)", "normal", c_darkgray),
        ("Top-1 Conf: max(p_t) (d=1)", "normal", c_darkgray),
        ("Margin: p_(1) - p_(2) (d=1)", "normal", c_darkgray),
        ("Action History: h_t in {0,1}^9 (d=9)", "normal", c_darkgray),
        ("Action Mask: m_t in {0,1}^9 (d=9)", "normal", c_darkgray),
        ("Decomposition: 7+1+1+1+9+9=28", "bold", c_orange)
    ], bg="#FFFFFF", border=c_darkgray, body_size=6.4)

    # 4. Multi-Head Policy Network (Exact Implemented Equation)
    draw_card(55.5, 73.0, 19.5, 18.0, "4. Multi-Head Policy", "MultiHeadActionPolicy", [
        ("Evaluates candidate a in {A1..A9}:", "bold", c_orange),
        ("Quality Head: dQ_hat in R", "normal", c_darkgray),
        ("Confidence Head: dC_hat in R", "normal", c_darkgray),
        ("Instability Head: logit_i in R", "normal", c_amber),
        ("Exact Implemented Utility Formula:", "bold", c_darkgray),
        ("U(s_t,a) = lambda_Q*dQ + lambda_C*dC", "bold", c_darkgray),
        (" - lambda_I*sigma(logit_i) - lambda_K*1.0", "bold", c_darkgray),
        ("Weights: [5.0, 1.0, 0.5, 0.25]", "normal", c_gray),
        ("Greedy: a* = argmax_{m[a]=1} U(s_t,a)", "bold", c_orange)
    ], bg="#FEEBC8", border=c_orange, body_size=6.3)

    # 5. Virtual-View Generator
    draw_card(77.0, 75.5, 19.0, 14.5, "5. Virtual View", "Deterministic Transform", [
        ("Apply a* to raw image x:", "bold", c_skyblue),
        ("Zoom, Crop, Contrast (A1..A9)", "normal", c_darkgray),
        ("Produces view: v_(t+1)", "bold", c_darkgray),
        ("Forward Pass:", "bold", c_blue),
        ("f_theta(v_(t+1)) -> p_new in Delta^6", "bold", c_darkgray)
    ], bg="#EBF8FF", border=c_skyblue)

    # Bottom Row of Panel A: Clockwise Return Flow
    # 6. Recursive Arithmetic Prediction Fusion
    draw_card(68.5, 53.5, 27.5, 16.5, "6. Recursive Arithmetic Fusion", "Correlated Running Mean", [
        ("Deterministic Recurrence Rule:", "bold", c_orange),
        ("p_(t+1) = (n_t * p_t + p_new) / (n_t + 1)", "bold", c_darkgray),
        ("n_(t+1) = n_t + 1  (with n_0 = 1)", "bold", c_darkgray),
        ("Strictly arithmetic mean of predictions", "normal", c_gray),
        ("*Correlated views derived from same 2D photograph*", "italic", c_darkgray),
        ("*No Bayesian update; no guaranteed variance reduction*", "italic", c_red)
    ], bg="#FEFCBF", border=c_orange, lw=1.4)

    # 7. Runtime Stopping Gate (Exact Source-Code Condition)
    draw_card(37.0, 53.5, 27.5, 16.5, "7. Runtime Stopping Gate", "Utility Threshold & Budget Limit", [
        ("Learned Stopping Evaluation:", "bold", c_green),
        ("1. Below Utility Threshold:", "bold", c_darkgray),
        ("   max_{m[a]=1} U(s_t,a) < tau  (tau = -0.4286)", "bold", c_green),
        ("   Reason: 'Below Utility Threshold'", "normal", c_gray),
        ("2. Budget Limit Exhausted:", "bold", c_darkgray),
        ("   Step b == 5  (B_max = 5 acquisitions)", "bold", c_darkgray),
        ("   Reason: 'Budget Exhausted'", "normal", c_gray),
        ("If Met: Terminate -> Output Payload", "bold", c_green),
        ("If Not Met: Continue active perception loop", "bold", c_orange),
        ("*Distinct from offline comparator CAL stopping*", "italic", c_gray)
    ], bg="#E6FFFA", border=c_green, lw=1.3, body_size=6.3)

    # 8. Final Fused Prediction Payload
    draw_card(3.5, 53.5, 31.5, 16.5, "8. Final Predicted Class & Trace", "Terminal Output Payload", [
        ("Terminal Prediction Output:", "bold", c_darkgray),
        ("Predicted Class: y_hat = argmax_c(p_T,c)", "bold", c_blue),
        ("Fused Probability Vector: p_T in Delta^6", "bold", c_darkgray),
        ("Executed Action Sequence: (A0, a_1*, ..., a_T*)", "normal", c_darkgray),
        ("Instability Warning: sigma(logit_i) in [0, 1]", "normal", c_amber),
        ("Terminal Reason: Below Utility vs Budget Exhausted", "normal", c_gray)
    ], bg="#FFFFFF", border=c_darkgray, lw=1.3)

    # Panel A Connectors
    draw_arrow((18.0, 82.5), (20.0, 82.5))  # 1 -> 2
    draw_arrow((35.0, 82.5), (37.0, 82.5))  # 2 -> 3
    draw_arrow((53.5, 82.5), (55.5, 82.5))  # 3 -> 4
    draw_arrow((75.0, 82.5), (77.0, 82.5))  # 4 -> 5
    draw_arrow((86.5, 75.5), (86.5, 70.0))  # 5 -> 6 (downwards)
    draw_arrow((68.5, 61.5), (64.5, 61.5))  # 6 -> 7 (leftwards)
    draw_arrow((37.0, 61.5), (35.0, 61.5))  # 7 -> 8 (leftwards on Stop)

    # Unobstructed Recurrent Feedback Arrow (Gate -> State Vector)
    # Goes straight UP vertically from (45.25, 70.0) to (45.25, 75.5)
    draw_arrow((45.25, 70.0), (45.25, 75.5), style=":", color=c_orange, lw=1.6)
    ax.text(46.0, 72.8, "Continue: s_(t+1)", fontsize=7.2, fontweight="bold", color=c_orange, zorder=5)

    # =========================================================================
    # PANEL B: GROUP-AWARE OFFLINE EVALUATION (FOUR DISTINCT ANALYSES)
    # =========================================================================
    # Overhead Clean Source-Data Distribution Bus
    # Horizontal bus bar at y=42.5 spanning from x=20.0 to x=88.0
    bus_y = 42.5
    ax.plot([20.0, 88.0], [bus_y, bus_y], "--", color=c_purple, lw=1.5, zorder=3)
    ax.text(54.0, bus_y + 1.0, "Offline Source-Data Distribution Bus  [Dataset Provenance & Partition Routing]",
            ha="center", va="center", fontsize=7.6, fontweight="bold", color=c_purple, zorder=3)

    # Left Input Card: Dataset Manifest & Precomputed Artifacts
    draw_card(3.5, 6.0, 16.5, 34.0, "ISIC 2019 Partitions", "13,931 Partitioning Groups", [
        ("Authoritative Group Census:", "bold", c_darkgray),
        ("11,847 observed lesion IDs", "normal", c_darkgray),
        ("+ 2,084 missing-ID singletons", "normal", c_darkgray),
        ("Four Group-Disjoint Splits:", "bold", c_darkgray),
        ("Train: N=16,702 (9,198 grp)", "normal", c_gray),
        ("Val: N=3,542 (1,972 grp)", "normal", c_gray),
        ("Cal: N=1,401 (789 grp)", "normal", c_gray),
        ("Test: N=3,686 (1,972 grp)", "bold", c_blue),
        ("Pairwise overlap = 0 (disjoint)", "italic", c_darkgray),
        ("Preserved Test Tensor:", "bold", c_darkgray),
        ("Precomputed Test Tensor", "bold", c_darkgray),
        ("10 actions cached offline", "normal", c_gray),
        ("Logits: (N, 10, 7)", "normal", c_gray),
        ("Embeddings: (N, 10, 1792)", "normal", c_gray)
    ], bg="#FFFFFF", border=c_darkgray, body_size=6.3)

    # Feed from Left Card into Overhead Bus
    # Vertical connector up from Left Card top (18.0, 40.0) up to bus (18.0, 42.5), then connects to bus
    ax.plot([18.0, 20.0], [bus_y, bus_y], "--", color=c_purple, lw=1.5, zorder=3)
    ax.plot([18.0, 18.0], [40.0, bus_y], "--", color=c_purple, lw=1.5, zorder=3)

    # Analysis 1: Conformal Calibration
    draw_card(22.0, 6.0, 17.0, 34.0, "Analysis 1: Conformal", "Split-Conformal Calibration", [
        ("Fitting Split D_fit (subset D_cal):", "bold", c_green),
        ("N_fit = 698 images", "normal", c_darkgray),
        ("K_fit = 393 grouping units", "normal", c_darkgray),
        ("NLL optimization on A0", "normal", c_gray),
        ("T_fit = 0.855690", "bold", c_green),
        ("Calibration Split D_conf (subset):", "bold", c_green),
        ("N_conf = 703 images", "normal", c_darkgray),
        ("K_conf = 396 grouping units", "normal", c_darkgray),
        ("Nonconformity: s = 1 - p_T(y|x)", "normal", c_darkgray),
        ("q* = 0.904267 (k = 634)", "bold", c_green),
        ("Test Evaluation (N=3,686):", "bold", c_darkgray),
        ("Observed Coverage: 90.6674%", "bold", c_green),
        ("Set Size: 1.9249 (3,342 / 3,686)", "normal", c_darkgray),
        ("*D_fit & D_conf partition D_cal*", "italic", c_gray),
        ("*Units = lesion_id + singleton IDs*", "italic", c_gray)
    ], bg="#E6FFFA", border=c_green, body_size=6.3)

    # Drop-down connector to Analysis 1 (clean vertical drop, zero card crossings!)
    draw_arrow((30.5, bus_y), (30.5, 40.0), style="--", color=c_green)

    # Analysis 2: Comparator Benchmark (Explicitly Shows CAL-calibrated Confidence Stopping)
    draw_card(41.0, 6.0, 17.0, 34.0, "Analysis 2: Comparator", "Offline Experimental Control", [
        ("Fixed Sequence Baseline:", "bold", c_red),
        ("A6 -> A5 -> A3 -> A1 -> A4", "bold", c_darkgray),
        ("Ranked on D_val (N=3,542)", "normal", c_gray),
        ("Comparator Stopping Calibration:", "bold", c_darkgray),
        ("Confidence Gate: max_c(p_c) >= tau_B", "bold", c_red),
        ("tau_B tuned on D_cal (N=1,401)", "normal", c_gray),
        ("Tested across budgets B in {1..5}", "normal", c_gray),
        ("Empirical Test Benchmark:", "bold", c_darkgray),
        ("No significant Top-1 difference", "bold", c_red),
        ("Exact McNemar test:", "normal", c_darkgray),
        ("p_adj >= 0.207 across all B", "bold", c_darkgray),
        ("Holm-Bonferroni adjusted", "normal", c_gray),
        ("*Statistical equivalence unestablished*", "italic", c_red),
        ("*tau_B stopping is comparator-only*", "italic", c_darkgray),
        ("*Shares D_cal with Analysis 1*", "italic", c_gray)
    ], bg="#FFF5F5", border=c_red, body_size=6.3)

    # Drop-down connector to Analysis 2 (clean vertical drop, zero card crossings!)
    draw_arrow((49.5, bus_y), (49.5, 40.0), style="--", color=c_red)

    # Analysis 3: Decision-Flip Audit
    draw_card(60.0, 6.0, 17.0, 34.0, "Analysis 3: Decision Flips", "Instability Head Evaluation", [
        ("Realized Test Transitions:", "bold", c_amber),
        ("M = 14,815 transitions", "bold", c_darkgray),
        ("Observed decision flips:", "normal", c_darkgray),
        ("988 flips (6.67% prev)", "normal", c_darkgray),
        ("Auxiliary Instability Head:", "bold", c_darkgray),
        ("Predicts flip risk logit_i", "normal", c_gray),
        ("AUROC: 0.947302", "bold", c_amber),
        ("AUPRC: 0.517218", "bold", c_darkgray),
        ("(Baseline prev: 0.0667)", "normal", c_gray),
        ("Uncertainty Methodology:", "bold", c_darkgray),
        ("Cluster Bootstrap (B=2,000)", "bold", c_blue),
        ("1,887 units (1,627 obs+260 sing)", "normal", c_darkgray),
        ("*On-policy pre-acquisition states*", "italic", c_gray),
        ("*1,887 multi-view test groups*", "italic", c_gray)
    ], bg="#FEFCBF", border=c_amber, body_size=6.3)

    # Drop-down connector to Analysis 3 (clean vertical drop, zero card crossings!)
    draw_arrow((68.5, bus_y), (68.5, 40.0), style="--", color=c_amber)

    # Analysis 4: Robustness Battery
    draw_card(79.0, 6.0, 17.5, 34.0, "Analysis 4: Robustness", "16 Digital Shift Conditions", [
        ("Controlled Perturbations:", "bold", c_skyblue),
        ("50 validation images evaluated", "bold", c_darkgray),
        ("16 digital shift conditions", "normal", c_darkgray),
        ("47 images with unique lesion IDs", "normal", c_darkgray),
        ("3 missing-ID singletons", "normal", c_darkgray),
        ("Distributional Shift Metric:", "bold", c_darkgray),
        ("Total Variation Dist. (TVD)", "bold", c_darkgray),
        ("Geometric vs photometric shifts", "normal", c_gray),
        ("Uncertainty Methodology:", "bold", c_darkgray),
        ("Paired Stratified", "bold", c_blue),
        ("Image-Level Bootstrap", "bold", c_blue),
        ("B = 1,000 (seed: 20261112)", "bold", c_darkgray),
        ("*Validation cohort; disjoint from test*", "italic", c_gray),
        ("*Quantifies algorithmic stability*", "italic", c_gray)
    ], bg="#EBF8FF", border=c_skyblue, body_size=6.3)

    # Drop-down connector to Analysis 4 (clean vertical drop, zero card crossings!)
    draw_arrow((87.5, bus_y), (87.5, 40.0), style="--", color=c_skyblue)

    # Legend at Bottom (inside Panel B, y=4.2)
    leg_y = 4.2
    ax.plot([14, 17], [leg_y, leg_y], "-", color=c_darkgray, lw=1.5)
    ax.text(17.5, leg_y, "Runtime Synchronous Execution", va="center", fontsize=6.8, color=c_darkgray)

    ax.plot([38, 41], [leg_y, leg_y], "--", color=c_purple, lw=1.5)
    ax.text(41.5, leg_y, "Offline Data Distribution Bus & Analysis Provenance (Not Statistically Independent)", va="center", fontsize=6.8, color=c_purple)

    ax.plot([78, 81], [leg_y, leg_y], ":", color=c_orange, lw=1.5)
    ax.text(81.5, leg_y, "Unobstructed Recurrent Feedback Loop", va="center", fontsize=6.8, color=c_orange)

    # Save to V3_2 directory
    out_dir = _FG_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / "fig1_architecture.svg"
    pdf_path = out_dir / "fig1_architecture.pdf"
    png_path = out_dir / "fig1_architecture.png"

    plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.1)
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"Successfully generated Figure 1 V3.2 deliverables:")
    print(f"  - SVG: {svg_path} ({svg_path.stat().st_size} bytes)")
    print(f"  - PDF: {pdf_path} ({pdf_path.stat().st_size} bytes)")
    print(f"  - PNG: {png_path} ({png_path.stat().st_size} bytes)")


if __name__ == "__main__":
    create_figure()
