"""Reproducible Graphical Abstract generator for BSPC submission (DERMA-ACT V2).

Complies strictly with Elsevier Graphical Abstract technical rules:
- Aspect ratio: 500 x 200 (2.5 : 1 landscape)
- Output resolution: 2500 x 1000 px at 300 dpi (exceeds 1328 x 531 px minimum)
- Font: Arial (Elsevier standard sans-serif)
- Zero "Graphical Abstract" heading text; zero GenAI artwork
- Strictly frozen ISIC 2019 V3.2 computational metrics only
"""

import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import argparse
_ap = argparse.ArgumentParser(description="Generate BSPC 2500x1000 px (300 dpi) Graphical Abstract.")
_ap.add_argument("--out-dir", type=pathlib.Path, default=None, help="Output directory")
_pa, _ = _ap.parse_known_args()
OUT_DIR = (_pa.out_dir or (pathlib.Path(__file__).resolve().parents[1] / "outputs" / "graphical_abstract")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, fc, ec, lw=1.3, radius=0.012, zorder=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.7, rad=0.0, zorder=4):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>,head_length=5.5,head_width=3.5",
        color=color,
        linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arr)


def main():
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42

    # Exact 2.5 : 1 ratio -> 2500 x 1000 px at 300 dpi
    fig = plt.figure(figsize=(2500 / 300.0, 1000 / 300.0), dpi=300, facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    BLUE = "#0072B2"
    TEAL = "#009E73"
    AMBER = "#E69F00"
    VERM = "#D55E00"
    SLATE = "#1E293B"
    MUTED = "#475569"

    # Outer zone containers (strict margins >= 0.015 all around)
    add_box(ax, 0.015, 0.165, 0.255, 0.80, "#F8FAFC", "#CBD5E1", lw=1.2, radius=0.016, zorder=1)
    add_box(ax, 0.290, 0.165, 0.365, 0.80, "#F0F7FC", "#93C5FD", lw=1.2, radius=0.016, zorder=1)
    add_box(ax, 0.675, 0.165, 0.310, 0.80, "#FFFBF5", "#FDBA74", lw=1.2, radius=0.016, zorder=1)

    # Concise Zone Headers well inside boundaries
    ax.text(0.1425, 0.922, "1. Input & Baseline", ha="center", va="center", fontsize=10.5, fontweight="bold", color=SLATE)
    ax.text(0.4725, 0.922, "2. Sequential Views & Flip Scoring", ha="center", va="center", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(0.8300, 0.922, "3. Uncertainty & Perturbations", ha="center", va="center", fontsize=10.5, fontweight="bold", color=VERM)

    # --- ZONE 1 ---
    add_box(ax, 0.030, 0.675, 0.225, 0.175, "#FFFFFF", SLATE, lw=1.3, radius=0.010)
    ax.text(0.1425, 0.798, "ISIC 2019 Test Input", ha="center", va="center", fontsize=9.2, fontweight="bold", color=SLATE)
    ax.text(0.1425, 0.732, "N = 3,686 (K = 1,972 lesion groups)\n7 classes (no patient IDs)", ha="center", va="center", fontsize=7.3, color=MUTED)

    add_arrow(ax, 0.1425, 0.670, 0.1425, 0.610, color=SLATE)

    add_box(ax, 0.030, 0.425, 0.225, 0.180, "#EFF6FF", BLUE, lw=1.4, radius=0.010)
    ax.text(0.1425, 0.550, "Frozen EfficientNet-B4", ha="center", va="center", fontsize=9.2, fontweight="bold", color=BLUE)
    ax.text(0.1425, 0.482, "Baseline view A0 prediction p0\nTop-1 Error: 30.79% (Acc: 69.21%)", ha="center", va="center", fontsize=7.3, color=SLATE)

    add_box(ax, 0.030, 0.195, 0.225, 0.175, "#FFFFFF", "#94A3B8", lw=1.1, radius=0.010)
    ax.text(0.1425, 0.318, "Virtual View Operators", ha="center", va="center", fontsize=8.8, fontweight="bold", color=SLATE)
    ax.text(0.1425, 0.252, "Crops A1-A6 | Gabor A7 (clipping)\nColor A8 | Dark-mask A9 (<=5 views)", ha="center", va="center", fontsize=7.0, color=MUTED)

    # Arrows from Zone 1 -> Zone 2
    add_arrow(ax, 0.258, 0.515, 0.305, 0.720, color=BLUE)
    add_arrow(ax, 0.258, 0.285, 0.305, 0.680, color=MUTED, rad=-0.12)

    # --- ZONE 2 ---
    add_box(ax, 0.308, 0.625, 0.329, 0.225, "#FFFFFF", BLUE, lw=1.3, radius=0.010)
    ax.text(0.4725, 0.805, "Sequential Selection & Fusion", ha="center", va="center", fontsize=9.2, fontweight="bold", color=BLUE)
    ax.text(
        0.4725,
        0.710,
        "Learned Policy V1 (A1-A9) vs Fixed Sequence\nArithmetic fusion: p_{t+1} = (n_t p_t + p_new) / (n_t + 1)\nEqual-inventory (no retrain): 26.21% vs 25.96%",
        ha="center",
        va="center",
        fontsize=7.2,
        color=SLATE,
    )

    add_arrow(ax, 0.4725, 0.620, 0.4725, 0.570, color=TEAL)

    add_box(ax, 0.308, 0.385, 0.329, 0.180, "#ECFDF5", TEAL, lw=1.3, radius=0.010)
    ax.text(0.4725, 0.520, "Flip Forecasting & Risk Stopping", ha="center", va="center", fontsize=9.2, fontweight="bold", color=TEAL)
    ax.text(
        0.4725,
        0.445,
        "Pre-view instability head (M = 14,815 transitions):\nAUROC = 0.9473 | AP = 0.5172 (base rate 6.67%)\nStop when utility < -0.4286, conf >= tau_B, or B = 5",
        ha="center",
        va="center",
        fontsize=7.1,
        color=SLATE,
    )

    add_box(ax, 0.308, 0.195, 0.329, 0.165, "#FFFFFF", "#94A3B8", lw=1.1, radius=0.010)
    ax.text(0.4725, 0.332, "Multi-View Error Audit (B = 5, p_adj = 0.319)", ha="center", va="center", fontsize=8.2, fontweight="bold", color=SLATE)
    bars = [
        ("A0 Baseline", 30.79, MUTED, 0.290),
        ("Learned V1", 26.94, BLUE, 0.252),
        ("Fixed Stop", 25.96, AMBER, 0.214),
    ]
    for label, val, col, yy in bars:
        ax.text(0.320, yy, label, ha="left", va="center", fontsize=7.2, fontweight="bold", color=SLATE)
        bw = (val / 35.0) * 0.165
        ax.barh(yy, bw, left=0.415, height=0.022, color=col, edgecolor="none", zorder=3)
        ax.text(0.420 + bw, yy, f"{val:.2f}%", ha="left", va="center", fontsize=7.2, fontweight="bold", color=col)

    # --- ZONE 3 ---
    add_arrow(ax, 0.640, 0.475, 0.688, 0.700, color=SLATE)

    add_box(ax, 0.692, 0.485, 0.276, 0.365, "#FFFFFF", VERM, lw=1.3, radius=0.010)
    ax.text(0.830, 0.808, "Split-Conformal Sets (alpha = 0.10)", ha="center", va="center", fontsize=9.0, fontweight="bold", color=VERM)
    ax.text(
        0.830,
        0.742,
        "Order statistic k* = 634 (q* = 0.904)\nMean size: 1.92 classes | 0% empty sets",
        ha="center",
        va="center",
        fontsize=7.2,
        color=MUTED,
    )

    # Put 90% reference line cleanly inside chart zone, with ALL numeric labels placed AFTER x=0.908 so nothing intersects line
    bar_left = 0.805
    bar_scale = 0.095  # 100% -> x=0.900, 90% -> x=0.8905
    target_x = bar_left + 0.90 * bar_scale
    ax.plot([target_x, target_x], [0.512, 0.662], ls="--", lw=1.1, color=SLATE, zorder=5)
    ax.text(target_x, 0.672, "90% target", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=SLATE)

    covs = [
        ("Marginal", 90.67, TEAL, 0.635),
        ("AKIEC Pooled", 74.21, VERM, 0.580),
        ("AKIEC Mondrian", 70.59, "#B91C1C", 0.525),
    ]
    for clabel, cval, ccol, cyy in covs:
        ax.text(0.703, cyy, clabel, ha="left", va="center", fontsize=7.1, fontweight="bold", color=SLATE)
        cw = (cval / 100.0) * bar_scale
        ax.barh(cyy, cw, left=bar_left, height=0.028, color=ccol, edgecolor="none", zorder=3)
        ax.text(0.906, cyy, f"{cval:.2f}%", ha="left", va="center", fontsize=7.1, fontweight="bold", color=ccol)

    add_box(ax, 0.692, 0.195, 0.276, 0.260, "#FFF7ED", AMBER, lw=1.3, radius=0.010)
    ax.text(0.830, 0.415, "Digital Perturbation Audit (A0)", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#9A3412")
    ax.text(
        0.830,
        0.300,
        "16 digital transforms (N = 50 val images):\n5-px shift / 5% crop: TVD 0.078 - 0.086\n+/-5% light / gamma: TVD 0.013 - 0.023\n(Isolated A0 test; not smartphone capture)",
        ha="center",
        va="center",
        fontsize=7.0,
        color=SLATE,
    )

    # Bottom disclaimer footer cleanly wrapped on 2 short centered lines
    add_box(ax, 0.015, 0.025, 0.970, 0.112, "#F1F5F9", "#94A3B8", lw=1.0, radius=0.010, zorder=2)
    ax.text(
        0.500,
        0.081,
        "Scope Boundary: Comparative retrospective computational audit on ISIC 2019 using frozen weights and offline artifacts.\nPatient-level independence, prospective clinical validation, expert reader testing, and multi-center generalization remain unverified.",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=SLATE,
    )

    stem = "graphical_abstract_bspc"
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{stem}.tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.svg")
    plt.close(fig)
    print("Regenerated Graphical Abstract assets cleanly in:", OUT_DIR)


if __name__ == "__main__":
    main()
