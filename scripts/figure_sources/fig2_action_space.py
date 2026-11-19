"""
Figure 2 Virtual-View Action Space & Image Transformation Visualization (Release V1.3).
DERMA-ACT Spatial Action Mapping (Panel A) & Canonical A0-A9 Virtual-View Outputs (Panel B).
Milestone P5.6.1 Action Quality Claim Reconciliation Release.

Generates:
  - fig2_virtual_view_action_space_v1_3.svg (Editable Vector SVG)
  - fig2_virtual_view_action_space_v1_3.pdf (Publication Vector PDF)
  - fig2_virtual_view_action_space_v1_3.png (300 DPI high-resolution raster)
"""
from pathlib import Path

import argparse as _fg_argparse
from pathlib import Path as _FgPath
_fg_ap = _fg_argparse.ArgumentParser(description='Figure 2: Virtual View Action Space (Tier B: requires local ISIC 2019 image & rendered views)')
_fg_ap.add_argument('--root', type=_FgPath, default=_FgPath(__file__).resolve().parents[2], help='Data/repository root')
_fg_ap.add_argument('--out-dir', type=_FgPath, default=_FgPath(__file__).resolve().parents[2] / 'outputs/figures', help='Output figure directory')
_fg_args, _ = _fg_ap.parse_known_args()
_FG_ROOT = _fg_args.root.resolve()
_FG_OUT = _fg_args.out_dir.resolve()
_FG_OUT.mkdir(parents=True, exist_ok=True)

import json
import cv2
import numpy as np
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
    c_blue = "#0072B2"      # Baseline Global View & Spatial crops
    c_cyan = "#56B4E9"      # Center Zoom
    c_amber = "#E69F00"     # Central-Mask Border View
    c_purple = "#7B2CBF"    # Gabor Texture Transformation (Saturated)
    c_green = "#009E73"     # Color Constancy Normalization
    c_red = "#C53030"       # Morphological Inpainting
    c_darkgray = "#2D3748"  # High-contrast borders & text
    c_gray = "#718096"      # Subtitles & secondary notes

    # Paths
    pilot_dir = _FG_ROOT / "data/figure2_views"
    v1_2_dir = pilot_dir / "V1_2"
    v1_3_dir = pilot_dir / "V1_3"
    v1_3_dir.mkdir(parents=True, exist_ok=True)
    views_dir = pilot_dir / "views"

    raw_path = _FG_ROOT / "data/isic2019/raw/ISIC_0012970_downsampled.jpg"
    if not raw_path.exists():
        alt_raw = _FG_ROOT / "data/isic2019/raw/ISIC_0012970.jpg"
        raw_path = alt_raw if alt_raw.exists() else raw_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing Tier B ISIC image for Figure 2: {raw_path}")
    raw_bgr = cv2.imread(str(raw_path))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = raw_rgb.shape[:2]

    # Overall Headers
    ax.text(50, 98.3, "DERMA-ACT: Virtual-View Action Space & Image Transformation Visualization",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#1A202C")
    ax.text(50, 96.8, "Spatial Action Mapping (Panel A) vs. Canonical A0–A9 Virtual-View Outputs (Panel B) for Illustrative Source Image ISIC_0012970 (NV)",
            ha="center", va="center", fontsize=9.0, color=c_gray)

    # =========================================================================
    # PANEL A: SPATIAL ACTION MAPPING
    # =========================================================================
    panel_a = patches.Rectangle((2, 51.5), 96, 43.5, facecolor="#F8FAFC", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax.add_patch(panel_a)

    ax.text(3.5, 93.3, "Panel A | Spatial / Geometric Action Coordinate Mapping on Source Lesion (ISIC_0012970, Melanocytic Nevus)",
            fontsize=10.5, fontweight="bold", color="#2D3748", zorder=3)

    # Panel A layout: 3 columns
    # Col 1: Centered source image with bounding boxes
    # Center lesion thumbnail
    img_ax = fig.add_axes([0.05, 0.54, 0.28, 0.36])
    img_ax.imshow(raw_rgb)
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    for spine in img_ax.spines.values():
        spine.set_edgecolor("#A0AEC0")
        spine.set_linewidth(1.0)

    # Overlay Bounding Boxes on Center Thumbnail
    # A1: Center crop (60% x 60%)
    ch, cw = h_orig // 2, w_orig // 2
    h_c, w_c = int(h_orig * 0.6), int(w_orig * 0.6)
    y1_a1, x1_a1 = ch - h_c // 2, cw - w_c // 2
    rect_a1 = patches.Rectangle((x1_a1, y1_a1), w_c, h_c, linewidth=2.0, edgecolor=c_cyan, facecolor="none", linestyle="--")
    img_ax.add_patch(rect_a1)

    # A2: Central mask (40% x 40%)
    h_m, w_m = int(h_orig * 0.4), int(w_orig * 0.4)
    y1_a2, x1_a2 = ch - h_m // 2, cw - w_m // 2
    rect_a2 = patches.Rectangle((x1_a2, y1_a2), w_m, h_m, linewidth=2.0, edgecolor=c_amber, facecolor="none", linestyle=":")
    img_ax.add_patch(rect_a2)

    # A3-A6 Directional Halves
    line_v = patches.ConnectionPatch((cw, 0), (cw, h_orig), "data", "data", axesA=img_ax, axesB=img_ax,
                                     color=c_blue, linewidth=1.5, linestyle="-.", alpha=0.85)
    line_h = patches.ConnectionPatch((0, ch), (w_orig, ch), "data", "data", axesA=img_ax, axesB=img_ax,
                                     color=c_blue, linewidth=1.5, linestyle="-.", alpha=0.85)
    img_ax.add_artist(line_v)
    img_ax.add_artist(line_h)

    # Legend & Labels for Panel A inside Col 2 & 3
    # Legend Box 1: Spatial & Geometric Actions (A1 - A6)
    leg1_box = patches.FancyBboxPatch((35.5, 53.5), 30.0, 37.0, boxstyle="round,pad=0.3,rounding_size=0.6",
                                      facecolor="#FFFFFF", edgecolor="#CBD5E0", linewidth=1.0, zorder=2)
    ax.add_patch(leg1_box)
    ax.text(37.0, 88.5, "Spatial Action Coordinate Systems", fontsize=9.2, fontweight="bold", color="#1A202C", zorder=3)

    coords_info = [
        (c_cyan, "--", "A1: Center Zoom (60% Lesion Core)", "Crop [0.20H:0.80H, 0.20W:0.80W] -> 224x224"),
        (c_amber, ":", "A2: Central-Mask Border View", "Center 40% masked to 0; peripheral pixels retained"),
        (c_blue, "-.", "A3: Upper Region View", "Crop top half [0:0.50H, 0:W] -> 224x224"),
        (c_blue, "-.", "A4: Lower Region View", "Crop bottom half [0.50H:H, 0:W] -> 224x224"),
        (c_blue, "-.", "A5: Left Region View", "Crop left half [0:H, 0:0.50W] -> 224x224"),
        (c_blue, "-.", "A6: Right Region View", "Crop right half [0:H, 0.50W:W] -> 224x224"),
    ]

    y_pos = 85.0
    for col_c, ls, name, desc in coords_info:
        ax.plot([37.0, 39.5], [y_pos, y_pos], color=col_c, linestyle=ls, linewidth=2.2, zorder=4)
        ax.text(40.5, y_pos + 0.4, name, fontsize=7.8, fontweight="bold", color="#2D3748", zorder=4)
        ax.text(40.5, y_pos - 1.6, desc, fontsize=6.8, color=c_gray, zorder=4)
        y_pos -= 5.4

    # Legend Box 2: Photometric, Frequency & Morphological Actions (A0, A7, A8, A9)
    leg2_box = patches.FancyBboxPatch((67.5, 53.5), 29.0, 37.0, boxstyle="round,pad=0.3,rounding_size=0.6",
                                      facecolor="#FFFFFF", edgecolor="#CBD5E0", linewidth=1.0, zorder=2)
    ax.add_patch(leg2_box)
    ax.text(69.0, 88.5, "Transformation Specifications", fontsize=9.2, fontweight="bold", color="#1A202C", zorder=3)

    non_spatial_info = [
        (c_blue, "A0: Baseline Global View", "Full unaugmented dermoscopic field; standardized 224x224"),
        (c_purple, "A7: Gabor Response — Numerical Saturation", "Multi-orientation Gabor filter (8-bit truncation before pooling)"),
        (c_green, "A8: Shades-of-Gray Color Constancy", "Illuminant normalization (Minkowski p=6); invariant skin tone"),
        (c_red, "A9: Dark-Structure Mask & Inpaint", "Morphological black-hat (17x17, T=10) + Telea inpainting"),
    ]

    y_pos = 85.0
    for col_c, name, desc in non_spatial_info:
        ax.scatter([69.2], [y_pos], color=col_c, s=45, zorder=4)
        ax.text(71.0, y_pos + 0.4, name, fontsize=7.8, fontweight="bold", color="#2D3748", zorder=4)
        ax.text(71.0, y_pos - 1.6, desc, fontsize=6.8, color=c_gray, zorder=4)
        y_pos -= 8.0

    # =========================================================================
    # PANEL B: CANONICAL 10-ACTION OUTPUT MANIFEST CARDS
    # =========================================================================
    panel_b = patches.Rectangle((2, 3.2), 96, 46.5, facecolor="#F8FAFC", edgecolor="#CBD5E0", linewidth=1.2, zorder=1)
    ax.add_patch(panel_b)

    ax.text(3.5, 48.0, "Panel B | Canonical 8-Bit Model Inputs (A0–A9) for Illustrative Case ISIC_0012970 (10 Action Candidates)",
            fontsize=10.5, fontweight="bold", color="#2D3748", zorder=3)

    with open(pilot_dir / "FIGURE_2_ACTION_OUTPUT_MANIFEST.json", "r") as f:
        action_manifest = json.load(f)

    # Layout for 10 cards: 2 rows of 5 cards
    col_xs = [3.5, 22.8, 42.1, 61.4, 80.7]
    card_w = 17.5
    card_h = 20.2

    action_configs = [
        # Action, Row, Col, Header Color, Title, Category Label, Detail Line 1, Detail Line 2, Detail Line 3
        ("A0", 0, 0, c_blue, "A0: Global View", "Baseline Global View", 
         "Full field 1.0x bilinear resize", "Standardized to 224 x 224;", "uncropped baseline reference"),
        ("A1", 0, 1, c_cyan, "A1: Center Zoom", "Spatial Transformation", 
         "Centered 60% x 60% spatial crop", "1.67x magnification factor;", "preserves raw RGB in crop window"),
        ("A2", 0, 2, c_amber, "A2: Central-Mask Border View", "Spatial Transformation", 
         "Central 40% masked to zero", "Peripheral image retained;", "no magnification or zoom"),
        ("A3", 0, 3, c_blue, "A3: Upper Region", "Spatial Transformation", 
         "Top half crop [0 : 0.5H]", "Resized to 224 x 224;", "original aspect ratio not preserved"),
        ("A4", 0, 4, c_blue, "A4: Lower Region", "Spatial Transformation", 
         "Bottom half crop [0.5H : H]", "Resized to 224 x 224;", "original aspect ratio not preserved"),
        ("A5", 1, 0, c_blue, "A5: Left Region", "Spatial Transformation", 
         "Left half crop [0 : 0.5W]", "Resized to 224 x 224;", "original aspect ratio not preserved"),
        ("A6", 1, 1, c_blue, "A6: Right Region", "Spatial Transformation", 
         "Right half crop [0.5W : W]", "Resized to 224 x 224;", "original aspect ratio not preserved"),
        ("A7", 1, 2, c_purple, "A7: Gabor Response", "Numerical Saturation", 
         "Pre-norm sat: 99.92% (p50: 99.97%)", "Canonical input: 99.92% at 255", "(25.69% cohort zero-collapse)"),
        ("A8", 1, 3, c_green, "A8: Shades-of-Gray", "Color Constancy Normalization", 
         "Shades-of-Gray (Minkowski p=6)", "Global RGB illuminant scaling;", "native spatial crop preserved"),
        ("A9", 1, 4, c_red, "A9: Dark-Structure Mask", "Morphological Inpainting", 
         "Black-hat (17x17, T=10) + Telea", "Illustrative: 49.44% mask (p99.1)", "Cohort: 8.13% median, 2365 comp"),
    ]

    for item in action_configs:
        tok, r, c, col, title_text, cat, line1, line2, line3 = item
        idx = int(tok[1:])
        row_data = action_manifest[idx]

        card_x = col_xs[c]
        card_y = 25.0 if r == 0 else 3.8

        # Draw outer card box
        card_box = patches.FancyBboxPatch(
            (card_x, card_y), card_w, card_h,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor="#FFFFFF", edgecolor=col, linewidth=1.2, zorder=2
        )
        ax.add_patch(card_box)

        # Header Badge
        badge_box = patches.FancyBboxPatch(
            (card_x + 0.5, card_y + card_h - 2.2), card_w - 1.0, 1.8,
            boxstyle="round,pad=0,rounding_size=0.4",
            facecolor=col, edgecolor="none", zorder=3
        )
        ax.add_patch(badge_box)
        ax.text(card_x + card_w/2, card_y + card_h - 1.3, title_text,
                ha="center", va="center", fontsize=6.6, fontweight="bold", color="#FFFFFF", zorder=4)

        # Action Image Thumbnail (Inset Axes inside the card)
        img_inset_x = (card_x + (card_w - 10.8) / 2) / 100.0
        img_inset_y = (card_y + 5.2) / 100.0
        img_inset_w = 10.8 / 100.0
        img_inset_h = 10.8 / 100.0

        ax_thumb = fig.add_axes([img_inset_x, img_inset_y, img_inset_w, img_inset_h])
        view_img = cv2.imread(str(pilot_dir / row_data["output_file"]))
        view_img = cv2.cvtColor(view_img, cv2.COLOR_BGR2RGB)
        ax_thumb.imshow(view_img)
        ax_thumb.set_xticks([])
        ax_thumb.set_yticks([])
        for spine in ax_thumb.spines.values():
            spine.set_edgecolor("#CBD5E0")
            spine.set_linewidth(0.8)

        # Special Display Inset for A7: Show display-only unclipped floating-point Gabor response
        if tok == "A7":
            # Float diagnostic inset axes in the top-right corner of the A7 thumbnail
            ax_gabor_inset = ax_thumb.inset_axes([0.48, 0.44, 0.50, 0.54])
            inset_img = cv2.imread(str(v1_3_dir / "A7_display_inset_unclipped.png"), cv2.IMREAD_GRAYSCALE)
            ax_gabor_inset.imshow(inset_img, cmap="gray")
            ax_gabor_inset.set_xticks([])
            ax_gabor_inset.set_yticks([])
            for spine in ax_gabor_inset.spines.values():
                spine.set_edgecolor("#FFD700")
                spine.set_linewidth(1.3)
            # Label the float diagnostic inset explicitly
            ax_thumb.text(0.98, 0.40, "Diagnostic float response\n(visualization only;\nnot classifier input)", 
                          color="#FFD700", fontsize=3.8, fontweight="bold",
                          ha="right", va="top", transform=ax_thumb.transAxes,
                          bbox=dict(boxstyle="round,pad=0.15", facecolor="#1A202C", alpha=0.92, edgecolor="#FFD700", lw=0.8))

        # Bottom text block inside card
        text_y_start = card_y + 4.2
        ax.text(card_x + card_w/2, text_y_start, line1,
                ha="center", va="center", fontsize=5.8, fontweight="bold", color="#2D3748", zorder=4)
        ax.text(card_x + card_w/2, text_y_start - 1.4, line2,
                ha="center", va="center", fontsize=5.4, color="#4A5568", zorder=4)
        ax.text(card_x + card_w/2, text_y_start - 2.8, line3,
                ha="center", va="center", fontsize=5.1, color=c_gray, zorder=4)

    # Footnote below Panel B
    ax.text(50, 1.4, "Note: ISIC_0012970 is an upper-tail illustrative case (A9 mask: 49.44%, cohort 99.1th percentile; cohort median: 8.13%, mean: 11.55%, 2,365 components).\nA7 intermediate pre-norm saturation (99.92%) reflects cohort median (99.97%), with post-norm zero-collapse to black in 25.69% of the validation cohort.",
            ha="center", va="center", fontsize=6.8, color="#4A5568", style="italic")

    # Save outputs in V1_3 directory
    out_svg = _FG_OUT / "fig2_action_space.svg"
    out_pdf = _FG_OUT / "fig2_action_space.pdf"
    out_png = _FG_OUT / "fig2_action_space.png"

    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated Figure 2 V1.3:")
    print(f"  SVG: {out_svg} ({out_svg.stat().st_size} bytes)")
    print(f"  PDF: {out_pdf} ({out_pdf.stat().st_size} bytes)")
    print(f"  PNG: {out_png} ({out_png.stat().st_size} bytes)")


if __name__ == "__main__":
    create_figure()
