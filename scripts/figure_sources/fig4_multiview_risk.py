"""
================================================================================
DERMA-ACT FIGURE 4 V1.1: MULTI-VIEW CLASSIFICATION RISK & VIRTUAL-VIEW ACQUISITION COST
Statistical Interpretation & Multi-Strategy Visual Correction Release.
Outputs:
  - fig4_multiview_risk_v1_1.pdf (vector)
  - fig4_multiview_risk_v1_1.svg (vector)
  - fig4_multiview_risk_v1_1.png (raster, 300 DPI)
================================================================================
"""

import os

import argparse as _fg_argparse
from pathlib import Path as _FgPath
_fg_ap = _fg_argparse.ArgumentParser(description='Figure 4: Multi-View Risk Reduction Benchmark (Tier A standalone)')
_fg_ap.add_argument('--root', type=_FgPath, default=_FgPath(__file__).resolve().parents[2], help='Data/repository root')
_fg_ap.add_argument('--out-dir', type=_FgPath, default=_FgPath(__file__).resolve().parents[2] / 'outputs/figures', help='Output figure directory')
_fg_args, _ = _fg_ap.parse_known_args()
_FG_ROOT = _fg_args.root.resolve()
_FG_OUT = _fg_args.out_dir.resolve()
_FG_OUT.mkdir(parents=True, exist_ok=True)

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ------------------------------------------------------------------------------
# 1. Matplotlib Publication Configuration
# ------------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.0,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.5,
    'lines.linewidth': 1.6,
    'lines.markersize': 6.0,
})

# Okabe-Ito Color Palette
COLOR_FIXED = '#0072B2'        # Blue
COLOR_FIXED_UNSTOP = '#56B4E9' # Sky Blue
COLOR_LEARNED = '#D55E00'      # Vermilion / Red-Orange
COLOR_GLOBAL = '#000000'       # Black
COLOR_NEUTRAL = '#718096'      # Gray
COLOR_BG_CARD = '#F8FAFC'      # Light Slate Background
COLOR_BORDER_CARD = '#E2E8F0'  # Light Border
COLOR_ALERT = '#C53030'        # Muted Red

# ------------------------------------------------------------------------------
# 2. Authoritative Data Loading
# ------------------------------------------------------------------------------
DATA_CSV = _FG_ROOT / "data/aggregate/FIGURE_4_V1_1_DATA.csv"

if not DATA_CSV.exists():
    raise FileNotFoundError(f"Required data file missing: {DATA_CSV}")

df_data = pd.read_csv(DATA_CSV)

row_a0 = df_data[df_data['strategy'] == 'Global_A0'].iloc[0]
df_v1 = df_data[df_data['strategy'] == 'Learned_V1'].sort_values('budget')
df_fixed_stop = df_data[df_data['strategy'] == 'Val_Ranked_ConfStop_Independent'].sort_values('budget')
df_fixed_unstop = df_data[df_data['strategy'] == 'Val_Ranked_Unstopped'].sort_values('budget')

# ------------------------------------------------------------------------------
# 3. Figure Layout Architecture
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(15.8, 11.2))

gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], width_ratios=[1.12, 1.0], wspace=0.22)
ax_c_plot = fig.add_subplot(gs_c[0, 0])
ax_c_card = fig.add_subplot(gs_c[0, 1])

# ------------------------------------------------------------------------------
# Panel A: Top-1 Misclassification Risk (0 - 35% Full Axis with Lower Inset Zoom)
# ------------------------------------------------------------------------------
budgets = np.array([0, 1, 2, 3, 4, 5])
risk_a0_pct = row_a0['risk'] * 100.0
ci_a0_l = row_a0['risk_ci95_lower'] * 100.0
ci_a0_u = row_a0['risk_ci95_upper'] * 100.0

v1_b = np.array([0] + df_v1['budget'].tolist())
v1_risk = np.array([risk_a0_pct] + (df_v1['risk'] * 100.0).tolist())
v1_ci_l = np.array([ci_a0_l] + (df_v1['risk_ci95_lower'] * 100.0).tolist())
v1_ci_u = np.array([ci_a0_u] + (df_v1['risk_ci95_upper'] * 100.0).tolist())

fix_b = np.array([0] + df_fixed_stop['budget'].tolist())
fix_risk = np.array([risk_a0_pct] + (df_fixed_stop['risk'] * 100.0).tolist())
fix_ci_l = np.array([ci_a0_l] + (df_fixed_stop['risk_ci95_lower'] * 100.0).tolist())
fix_ci_u = np.array([ci_a0_u] + (df_fixed_stop['risk_ci95_upper'] * 100.0).tolist())

unstop_b = np.array([0] + df_fixed_unstop['budget'].tolist())
unstop_risk = np.array([risk_a0_pct] + (df_fixed_unstop['risk'] * 100.0).tolist())
unstop_ci_l = np.array([ci_a0_l] + (df_fixed_unstop['risk_ci95_lower'] * 100.0).tolist())
unstop_ci_u = np.array([ci_a0_u] + (df_fixed_unstop['risk_ci95_upper'] * 100.0).tolist())

# Main Plot: Full 0 to 35% Scale
ax_a.axhline(risk_a0_pct, color=COLOR_GLOBAL, linestyle='--', linewidth=1.1, alpha=0.6, zorder=1)
ax_a.plot([0], [risk_a0_pct], marker='D', markersize=6.5, color=COLOR_GLOBAL, 
          markerfacecolor=COLOR_GLOBAL, markeredgecolor='white', markeredgewidth=1.0,
          linestyle='None', label='Single Global View ($A_0$, $B=0$)', zorder=6)

ax_a.plot(unstop_b, unstop_risk, color=COLOR_FIXED_UNSTOP, linestyle=':', 
          marker='^', markersize=5.5, markerfacecolor='white', markeredgewidth=1.4,
          label='Fixed Sequence (Unstopped)', zorder=3)

ax_a.plot(fix_b, fix_risk, color=COLOR_FIXED, linestyle='-', 
          marker='o', markersize=6.0, markerfacecolor=COLOR_FIXED, markeredgecolor='white', markeredgewidth=1.0,
          label='Fixed Sequence (CAL-Stop)', zorder=4)

ax_a.plot(v1_b, v1_risk, color=COLOR_LEARNED, linestyle='-', 
          marker='s', markersize=6.0, markerfacecolor=COLOR_LEARNED, markeredgecolor='white', markeredgewidth=1.0,
          label='Learned Policy V1 (Adaptive)', zorder=5)

ax_a.text(0.12, risk_a0_pct + 0.8, f"Single View $A_0$ Baseline ({risk_a0_pct:.2f}%)", 
          fontsize=8.0, fontweight='bold', color=COLOR_GLOBAL)

# Highlight zoom region on main axis
zoom_rect = Rectangle((-0.2, 23.5), 5.5, 9.0, fill=True, facecolor='#EDF2F7', edgecolor='#A0AEC0', 
                      linestyle='--', linewidth=0.9, alpha=0.5, zorder=0)
ax_a.add_patch(zoom_rect)
ax_a.text(5.25, 24.0, "Zoom Region (23.5–32.5%)", fontsize=7.0, fontstyle='italic', color='#718096', ha='right', va='bottom')

ax_a.set_title(r"$\mathbf{A}$   Top-1 Classification Risk across Budgets ($0-35\%$ Full Scale)", loc='left', pad=10)
ax_a.set_xlabel("Sequential Acquisition Budget ($B$)")
ax_a.set_ylabel("Top-1 Misclassification Risk (%)")
ax_a.set_xlim(-0.25, 5.5)
ax_a.set_ylim(0.0, 36.0)
ax_a.set_xticks(range(6))
ax_a.grid(True, linestyle=':', alpha=0.5)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# Legend placed compactly in lower left
ax_a.legend(loc='lower left', bbox_to_anchor=(0.02, 0.10), frameon=True, framealpha=0.95, 
            edgecolor=COLOR_BORDER_CARD, facecolor='white', fontsize=7.8)

ax_a.text(0.02, 0.025, "* All multi-view budgets significantly outperform $A_0$ ($p_{\\mathrm{adj}} \\leq 1.64 \\times 10^{-4}$)",
          transform=ax_a.transAxes, fontsize=7.4, fontstyle='italic', color='#4A5568')

# Inset Zoom Axes (23.5% to 32.5% focused view) positioned in the lower right
ax_ins = inset_axes(ax_a, width="48%", height="44%", loc='lower right', 
                    bbox_to_anchor=(0.0, 0.08, 0.98, 0.92), bbox_transform=ax_a.transAxes)

# Horizontal reference in inset
ax_ins.axhline(risk_a0_pct, color=COLOR_GLOBAL, linestyle='--', linewidth=1.1, alpha=0.6, zorder=1)
ax_ins.plot([0], [risk_a0_pct], marker='D', markersize=6.5, color=COLOR_GLOBAL, 
            markerfacecolor=COLOR_GLOBAL, markeredgecolor='white', markeredgewidth=1.0, linestyle='None', zorder=6)

# Horizontal dodge in inset zoom: Unstopped: -0.06, CAL-Stop: 0.0, Learned V1: +0.06
dodge_unstop = np.array([0.0] + (np.arange(1, 6) - 0.06).tolist())
dodge_fix = np.array([0.0] + np.arange(1, 6).tolist())
dodge_v1 = np.array([0.0] + (np.arange(1, 6) + 0.06).tolist())

ax_ins.fill_between(dodge_fix[1:], fix_ci_l[1:], fix_ci_u[1:], color=COLOR_FIXED, alpha=0.12, zorder=2)
ax_ins.fill_between(dodge_v1[1:], v1_ci_l[1:], v1_ci_u[1:], color=COLOR_LEARNED, alpha=0.15, zorder=2)

ax_ins.plot(dodge_unstop, unstop_risk, color=COLOR_FIXED_UNSTOP, linestyle=':', 
            marker='^', markersize=5.5, markerfacecolor='white', markeredgewidth=1.4, zorder=3)

ax_ins.plot(dodge_fix, fix_risk, color=COLOR_FIXED, linestyle='-', 
            marker='o', markersize=6.0, markerfacecolor=COLOR_FIXED, markeredgecolor='white', markeredgewidth=1.0, zorder=4)

ax_ins.plot(dodge_v1, v1_risk, color=COLOR_LEARNED, linestyle='-', 
            marker='s', markersize=6.0, markerfacecolor=COLOR_LEARNED, markeredgecolor='white', markeredgewidth=1.0, zorder=5)

for b_idx in range(1, 6):
    ax_ins.text(b_idx, 24.2, '*', ha='center', va='center', fontsize=12, fontweight='bold', color='#B91C1C')

ax_ins.set_title("Inset Zoom (23.5–32.5%) — Pointwise 95% CIs", fontsize=8.0, fontweight='bold', pad=4, color='#2D3748')
ax_ins.set_xlim(-0.2, 5.3)
ax_ins.set_ylim(23.5, 32.5)
ax_ins.set_xticks(range(6))
ax_ins.tick_params(axis='both', which='major', labelsize=7.0)
ax_ins.grid(True, linestyle=':', alpha=0.5)
ax_ins.spines['top'].set_visible(False)
ax_ins.spines['right'].set_visible(False)

# ------------------------------------------------------------------------------
# Panel B: Realized Virtual-View Acquisition Cost across Budgets
# ------------------------------------------------------------------------------
cost_unstop = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
v1_costs = np.array([0.0] + df_v1['mean_cost'].tolist())
fix_costs = np.array([0.0] + df_fixed_stop['mean_cost'].tolist())

# Nominal ceiling line (slope = 1)
ax_b.plot(budgets, cost_unstop, color=COLOR_NEUTRAL, linestyle='--', linewidth=1.2, 
          label='Nominal Budget Limit (Unstopped, $y=B$)', zorder=1)

ax_b.plot(budgets, fix_costs, color=COLOR_FIXED, linestyle='-', marker='o', markersize=6.0,
          markerfacecolor=COLOR_FIXED, markeredgecolor='white', markeredgewidth=1.0,
          label='Validation-Ranked Fixed Sequence (CAL-Stop)', zorder=3)

ax_b.plot(budgets, v1_costs, color=COLOR_LEARNED, linestyle='-', marker='s', markersize=6.0,
          markerfacecolor=COLOR_LEARNED, markeredgecolor='white', markeredgewidth=1.0,
          label='Learned Policy V1 (Adaptive)', zorder=4)

ax_b.plot([0], [0.0], marker='D', markersize=6.5, color=COLOR_GLOBAL, 
          markerfacecolor=COLOR_GLOBAL, markeredgecolor='white', markeredgewidth=1.0,
          linestyle='None', label='Single Global View ($A_0$, cost = 0.0)', zorder=5)

# Value callouts at B=5
ax_b.annotate(f"Fixed: {fix_costs[5]:.2f} views\n(-22.0% cost)",
             xy=(5, fix_costs[5]), xytext=(4.10, 3.10),
             arrowprops=dict(arrowstyle="->", color=COLOR_FIXED, lw=1.2),
             ha='center', va='top',
             fontsize=8.0, fontweight='bold', color=COLOR_FIXED,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLOR_BORDER_CARD, alpha=0.95))

ax_b.annotate(f"Learned: {v1_costs[5]:.2f} views\n(-19.6% cost)",
             xy=(5, v1_costs[5]), xytext=(4.15, 4.85),
             arrowprops=dict(arrowstyle="->", color=COLOR_LEARNED, lw=1.2),
             ha='center', va='bottom',
             fontsize=8.0, fontweight='bold', color=COLOR_LEARNED,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLOR_BORDER_CARD, alpha=0.95))

# Rigorous computational proxy qualification (Section 6)
ax_b.text(0.04, 0.035, "Note: Realized virtual-view count is a computational acquisition-cost proxy,\nnot wall-clock latency, energy consumption or physical camera repositioning time.",
          transform=ax_b.transAxes, fontsize=7.8, fontstyle='italic', color='#4A5568')

ax_b.set_title(r"$\mathbf{B}$   Realized Virtual-View Acquisition Cost across Budgets", loc='left', pad=10)
ax_b.set_xlabel("Sequential Acquisition Budget ($B$)")
ax_b.set_ylabel("Realized Virtual Views (Computational Proxy)")
ax_b.set_xlim(-0.25, 5.85)
ax_b.set_ylim(-0.6, 5.6)
ax_b.set_xticks(range(6))
ax_b.grid(True, linestyle=':', alpha=0.6)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor=COLOR_BORDER_CARD, facecolor='white', fontsize=8.0)

# ------------------------------------------------------------------------------
# Panel C (Left): Paired Difference Forest Plot (V1 vs Fixed CAL-Stop)
# ------------------------------------------------------------------------------
paired_b = df_v1['budget'].to_numpy()
paired_diff = df_v1['paired_risk_diff'].to_numpy() * 100.0
paired_ci_l = df_v1['paired_ci95_lower'].to_numpy() * 100.0
paired_ci_u = df_v1['paired_ci95_upper'].to_numpy() * 100.0
p_mcnemar = df_v1['mcnemar_p_value'].to_numpy()
p_holm = df_v1['holm_adjusted_p_value'].to_numpy()

# Zero reference line
ax_c_plot.axhline(0.0, color='#4A5568', linestyle='--', linewidth=1.2, zorder=1)

# Shaded background regions
ax_c_plot.axhspan(0.0, 2.7, color='#EBF8FF', alpha=0.6, zorder=0)
ax_c_plot.axhspan(-1.6, 0.0, color='#FFF5F5', alpha=0.6, zorder=0)

# Directional banner labels
ax_c_plot.text(0.03, 0.94, r"$\mathbf{\uparrow\ Favors\ Fixed\ Comparator}$ (Fixed sequence achieves lower risk)", 
               transform=ax_c_plot.transAxes, color=COLOR_FIXED, fontsize=8.5, fontweight='bold', va='top')
ax_c_plot.text(0.03, 0.09, r"$\mathbf{\downarrow\ Favors\ Learned\ Policy\ V1}$ (Learned policy achieves lower risk)", 
               transform=ax_c_plot.transAxes, color=COLOR_LEARNED, fontsize=8.5, fontweight='bold', va='bottom')

# Error bars (Pointwise 95% Grouped Bootstrap CIs)
yerr = np.array([paired_diff - paired_ci_l, paired_ci_u - paired_diff])
ax_c_plot.errorbar(paired_b, paired_diff, yerr=yerr, fmt='none', 
                   ecolor='#2D3748', elinewidth=1.6, capsize=4.5, capthick=1.4, zorder=3)

ax_c_plot.scatter(paired_b, paired_diff, s=75, c=COLOR_LEARNED, edgecolors='#2D3748', 
                  linewidth=1.2, zorder=4, label='Paired Point Estimate ($\Delta$Risk)')

# Numeric annotations above each point
for b_val, diff_val, l_val, u_val, p_m, p_h in zip(paired_b, paired_diff, paired_ci_l, paired_ci_u, p_mcnemar, p_holm):
    sign_str = "+" if diff_val > 0 else ""
    p_adj_str = f"{p_h:.3f}" if p_h < 1.0 else "1.000"
    label_txt = f"{sign_str}{diff_val:.2f}%\n" + r"$p_{\mathrm{adj}}=" + p_adj_str + "$"
    y_pos = u_val + 0.14
    ax_c_plot.text(b_val, y_pos, label_txt, ha='center', va='bottom', fontsize=7.5,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#CBD5E0', alpha=0.92))

# Multiplicity & Pointwise CI Clarification Note (Section 3) - Wrapped
ax_c_plot.text(0.03, 0.02, "Note: Intervals are pointwise and are not multiplicity-adjusted; p-values are Holm-adjusted (m=25).\nPointwise CI excluding zero at B=5 does not establish statistical significance after Holm correction.",
               transform=ax_c_plot.transAxes, fontsize=6.8, fontstyle='italic', color='#4A5568')

ax_c_plot.set_title(r"$\mathbf{C}$   Paired Strategy Comparison ($\Delta\mathrm{Risk} = \mathrm{Risk}_{\mathrm{Learned}} - \mathrm{Risk}_{\mathrm{Fixed}}$)", loc='left', pad=10)
ax_c_plot.set_xlabel("Sequential Acquisition Budget ($B$)")
ax_c_plot.set_ylabel(r"Paired Risk Difference ($\Delta$Risk, %)")
ax_c_plot.set_xlim(0.5, 5.5)
ax_c_plot.set_ylim(-1.6, 2.7)
ax_c_plot.set_xticks(range(1, 6))
ax_c_plot.grid(True, linestyle=':', alpha=0.6)
ax_c_plot.spines['top'].set_visible(False)
ax_c_plot.spines['right'].set_visible(False)

# ------------------------------------------------------------------------------
# Panel C (Right): Statistical Audit Card & Action Disparity Breakdown
# ------------------------------------------------------------------------------
ax_c_card.axis('off')

card_bbox = FancyBboxPatch((0.005, 0.005), 0.99, 0.99,
                           boxstyle="round,pad=0.01,rounding_size=0.025",
                           facecolor=COLOR_BG_CARD, edgecolor=COLOR_BORDER_CARD,
                           linewidth=1.2, transform=ax_c_card.transAxes)
ax_c_card.add_patch(card_bbox)

# Header
ax_c_card.text(0.04, 0.95, "Paired Statistical Tests & Action Inventory Disparity",
               fontsize=10.0, fontweight='bold', color='#1A202C', transform=ax_c_card.transAxes)

# Section 1: Exact Paired Statistical Results Table
y_cursor = 0.895
headers = ["Budget", "ΔRisk (%)", "Pointwise 95% CI", "Discordant (a/b)", "McNemar p", "Holm p_adj"]
col_x = [0.04, 0.16, 0.30, 0.55, 0.74, 0.88]

for h, x in zip(headers, col_x):
    ax_c_card.text(x, y_cursor, h, fontsize=7.5, fontweight='bold', color='#4A5568', transform=ax_c_card.transAxes)

y_cursor -= 0.015
ax_c_card.plot([0.04, 0.96], [y_cursor, y_cursor], color='#CBD5E0', linewidth=0.8, transform=ax_c_card.transAxes)
y_cursor -= 0.035

for _, r in df_v1.iterrows():
    b_int = int(r['budget'])
    diff_pct = r['paired_risk_diff'] * 100.0
    ci_str = f"[{r['paired_ci95_lower']*100:.2f}, {r['paired_ci95_upper']*100:.2f}]"
    disc_str = f"{int(r['discordant_v1_correct_comp_wrong'])} / {int(r['discordant_v1_wrong_comp_correct'])}"
    p_mc = f"{r['mcnemar_p_value']:.4f}"
    p_adj_str = f"{r['holm_adjusted_p_value']:.4f}" if r['holm_adjusted_p_value'] < 1.0 else "1.000"
    
    ax_c_card.text(col_x[0], y_cursor, f"B = {b_int}", fontsize=7.5, color='#2D3748', transform=ax_c_card.transAxes)
    ax_c_card.text(col_x[1], y_cursor, f"+{diff_pct:.2f}%", fontsize=7.5, fontweight='bold', color=COLOR_FIXED, transform=ax_c_card.transAxes)
    ax_c_card.text(col_x[2], y_cursor, ci_str, fontsize=7.5, color='#4A5568', transform=ax_c_card.transAxes)
    ax_c_card.text(col_x[3], y_cursor, disc_str, fontsize=7.5, color='#4A5568', transform=ax_c_card.transAxes)
    ax_c_card.text(col_x[4], y_cursor, p_mc, fontsize=7.5, color='#4A5568', transform=ax_c_card.transAxes)
    ax_c_card.text(col_x[5], y_cursor, p_adj_str, fontsize=7.5, fontweight='bold', color='#2D3748', transform=ax_c_card.transAxes)
    y_cursor -= 0.035

# Table bottom divider
y_cursor -= 0.005
ax_c_card.plot([0.04, 0.96], [y_cursor, y_cursor], color='#CBD5E0', linewidth=0.8, transform=ax_c_card.transAxes)

# Strict replacement of statistical parity language (Section 2)
y_cursor -= 0.022
verdict_text = (
    "Test Verdict: No statistically significant difference in Top-1 misclassification risk was detected\n"
    "after Holm correction ($p_{\\mathrm{adj}} \\geq 0.3185$); statistical equivalence was not established."
)
ax_c_card.text(0.04, y_cursor, verdict_text, fontsize=7.2, fontweight='bold', color='#2B6CB0', va='top', transform=ax_c_card.transAxes)

# Post-verdict divider line placed with generous clearance below line 2
y_cursor -= 0.068
ax_c_card.plot([0.04, 0.96], [y_cursor, y_cursor], color='#CBD5E0', linewidth=0.8, transform=ax_c_card.transAxes)

# Section 2: Critical Action-Composition Limitation (Section 7)
y_cursor -= 0.030
ax_c_card.text(0.04, y_cursor, "Critical Action-Composition Limitation (Non-Causal Disclosure):",
               fontsize=8.3, fontweight='bold', color='#C53030', transform=ax_c_card.transAxes)

limitation_text = (
    "• Validation-Ranked Fixed Comparator sequence strictly executes spatial actions:\n"
    "    $A_6$ (Right) $\\to A_5$ (Left) $\\to A_3$ (Upper) $\\to A_1$ (Center Zoom) $\\to A_4$ (Lower).\n"
    "    Zero access to central-mask border ($A_2$), Gabor ($A_7$), color ($A_8$), or inpainting ($A_9$).\n"
    "• Learned Policy V1 dynamically selects across full inventory ($A_1-A_9$):\n"
    "    Frequently deploys color constancy $A_8$ in 74.96% of cases and morphological dark-structure\n"
    "    masking and Telea inpainting $A_9$ in 20.51% of cases (with $A_7$ in 0.16% and $A_2$ in 3.42%).\n"
    "• Epistemological Conclusion: This benchmark compares two composite strategies with different\n"
    "    action inventories. Any risk difference reflects the combined effect of action composition and\n"
    "    policy routing. The analysis does not isolate adaptive routing as a causal factor."
)

y_cursor -= 0.024
ax_c_card.text(0.04, y_cursor, limitation_text, fontsize=7.2, color='#2D3748', va='top', transform=ax_c_card.transAxes)

# ------------------------------------------------------------------------------
# 4. Save Figures (PDF, SVG, PNG)
# ------------------------------------------------------------------------------
out_dir = _FG_OUT
pdf_path = out_dir / "fig4_multiview_risk.pdf"
svg_path = out_dir / "fig4_multiview_risk.svg"
png_path = out_dir / "fig4_multiview_risk.png"

plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
plt.savefig(svg_path, bbox_inches='tight', dpi=300)
plt.savefig(png_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"[SUCCESS] Exported PDF: {pdf_path}")
print(f"[SUCCESS] Exported SVG: {svg_path}")
print(f"[SUCCESS] Exported PNG: {png_path} (300 DPI)")
