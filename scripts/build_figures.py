"""Clean-room orchestrator to regenerate DERMA-ACT publication figures.

By default (--tier aggregate), reproduces all Tier A standalone figures that require
zero raw ISIC lesion images or restricted row-level prediction caches:
  - Figure 1 (fig1_architecture)
  - Figure 4 (fig4_multiview_risk)
  - Graphical Abstract (graphical_abstract_bspc)

Pass --tier all --root <TIER_B_DATA_ROOT> to generate Figures 1-8 when the Tier B
restricted dataset / frozen prediction bundle is mounted locally.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_step(script_rel: str, root: Path, out_dir: Path) -> None:
    script = REPO_ROOT / script_rel
    print(f"Running {script_rel}...")
    cmd = [sys.executable, str(script), "--out-dir", str(out_dir)]
    if script_rel != "scripts/generate_graphical_abstract.py":
        cmd.extend(["--root", str(root)])
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"Figure script failed: {script_rel}")
    last = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "done"
    print(f"  OK: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DERMA-ACT figures (Tier A standalone or full Tier B).")
    ap.add_argument("--tier", choices=["aggregate", "all"], default="aggregate")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="Workspace root holding Tier B artifacts (for --tier all)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = args.root.resolve()

    tier_a = [
        "scripts/figure_sources/fig1_architecture.py",
        "scripts/figure_sources/fig4_multiview_risk.py",
    ]
    tier_b = [
        "scripts/figure_sources/fig2_action_space.py",
        "scripts/figure_sources/fig3_sequential_trajectory.py",
        "scripts/figure_sources/fig5_class_specific_tradeoffs.py",
        "scripts/figure_sources/fig6_decision_flip_prediction.py",
        "scripts/figure_sources/fig7_conformal_uncertainty.py",
        "scripts/figure_sources/fig8_perturbation_robustness.py",
    ]

    for rel in tier_a:
        run_step(rel, root, out_dir)
    ga_dir = out_dir.parent / "graphical_abstract"
    run_step("scripts/generate_graphical_abstract.py", root, ga_dir)

    if args.tier == "all":
        for rel in tier_b:
            run_step(rel, root, out_dir)
    print(f"Completed --tier {args.tier} output in {out_dir}")


if __name__ == "__main__":
    main()
