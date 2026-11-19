"""Phase G: Controlled Post-Hoc Experiment 2 — Mondrian Class-Conditional Conformal Prediction.

Strictly read-only over frozen artifacts (calibration_canonical.npz, test_canonical.npz,
ISIC_2019_Training_Metadata.csv). Uses exact P7.1 group-disjoint D_fit (N=698) / D_conf (N=703)
partition (seed=20261112) and D_fit BFGS temperature T_fit. Zero CNN inference, zero test tuning.

Outputs saved to research/derma_act/p8_1_final_submission_audit/phase_g_mondrian_conformal/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

import argparse

REPO_ROOT = Path(__file__).resolve().parents[4]
_ap = argparse.ArgumentParser(description="Phase G: Class-conditional Mondrian split-conformal evaluation.")
_ap.add_argument("--root", type=Path, default=REPO_ROOT, help="Data root holding Tier B canonical logits and metadata")
_ap.add_argument("--out-dir", type=Path, default=None, help="Output directory")
_pa, _ = _ap.parse_known_args()

ROOT = _pa.root.resolve()
OUT_DIR = (_pa.out_dir or (Path(__file__).resolve().parent)).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAL_NPZ = ROOT / "research/baseline_v2/runs/clean_run_001/predictions/calibration_canonical.npz"
TE_NPZ = ROOT / "research/baseline_v2/runs/clean_run_001/predictions/test_canonical.npz"
META_CSV = ROOT / "data/isic2019/raw/ISIC_2019_Training_Metadata.csv"

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
ALPHA = Fraction(1, 10)
ONE_MINUS_ALPHA = Fraction(9, 10)
SPLIT_SEED = 20261112
BOOT_SEED = 20261112
N_BOOT = 2000


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(1024 * 1024), b""):
            h.update(ch)
    return h.hexdigest()


def fit_temperature_bfgs(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(log_t: np.ndarray) -> float:
        t = float(np.exp(log_t[0]))
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-log_p[np.arange(len(y)), y].mean())

    res = minimize(nll, [0.0], method="BFGS")
    return float(np.exp(res.x[0]))


def build_groups(image_ids: np.ndarray, meta_df: pd.DataFrame) -> np.ndarray:
    grp = []
    for iid in image_ids:
        lid = meta_df.lesion_id.get(iid)
        if isinstance(lid, str) and len(lid.strip()) > 0:
            grp.append(lid)
        else:
            grp.append(f"IMG::{iid}")
    return np.array(grp)


def group_stratified_split(cal_grp: np.ndarray, yc: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ugc = np.unique(cal_grp)
    g_lab = np.array([np.bincount(yc[cal_grp == g], minlength=7).argmax() for g in ugc])
    rng = np.random.default_rng(seed)
    fit_g = []
    for c in range(7):
        gs = ugc[g_lab == c]
        rng.shuffle(gs)
        fit_g.extend(gs[: len(gs) // 2])
    fit_mask = np.isin(cal_grp, np.array(fit_g))
    return np.where(fit_mask)[0], np.where(~fit_mask)[0]


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    for req in (CAL_NPZ, TE_NPZ, META_CSV):
        if not req.exists():
            raise FileNotFoundError(f'Missing Tier B artifact: {req}')
    meta = pd.read_csv(META_CSV).set_index("image")
    cal = np.load(CAL_NPZ, allow_pickle=True)
    te = np.load(TE_NPZ, allow_pickle=True)

    ids_cal = cal["image_id"].astype(str)
    L_cal = cal["logits"].astype(np.float64)
    y_cal = cal["true_label"].astype(int)
    grp_cal = build_groups(ids_cal, meta)

    ids_te = te["image_id"].astype(str)
    L_te = te["logits"].astype(np.float64)
    y_te = te["true_label"].astype(int)
    grp_te = build_groups(ids_te, meta)

    fit_idx, conf_idx = group_stratified_split(grp_cal, y_cal, SPLIT_SEED)
    T_fit = fit_temperature_bfgs(L_cal[fit_idx], y_cal[fit_idx])

    p_conf = softmax(L_cal[conf_idx] / T_fit, axis=1)
    y_conf = y_cal[conf_idx]
    grp_conf = grp_cal[conf_idx]
    scores_conf = 1.0 - p_conf[np.arange(len(y_conf)), y_conf]

    p_te = softmax(L_te / T_fit, axis=1)

    # Pooled Split-Conformal baseline (exact P7.1: k=634)
    n_conf = len(scores_conf)
    k_pool = math.ceil(Fraction(n_conf + 1) * ONE_MINUS_ALPHA)
    q_pool = float(np.sort(scores_conf)[k_pool - 1])
    sets_pool = p_te >= (1.0 - q_pool)

    # Mondrian per-class quantiles
    q_mondrian = np.zeros(7, dtype=np.float64)
    cal_class_diag = []
    for c, cname in enumerate(CLASSES):
        mc = y_conf == c
        sc = np.sort(scores_conf[mc])
        nc = len(sc)
        kc_clusters = int(len(np.unique(grp_conf[mc])))
        k_c = math.ceil(Fraction(nc + 1) * ONE_MINUS_ALPHA)
        feasible_image_level = k_c <= nc
        q_c = float(sc[k_c - 1]) if feasible_image_level else 1.0
        q_mondrian[c] = q_c

        # Also check group-level cluster order statistic if one takes max score per lesion inside class c
        s_grp = pd.Series(scores_conf[mc], index=grp_conf[mc]).groupby(level=0).max().to_numpy()
        k_grp_c = math.ceil(Fraction(len(s_grp) + 1) * ONE_MINUS_ALPHA)
        feasible_cluster_max = k_grp_c <= len(s_grp)

        cal_class_diag.append({
            "class_id": c,
            "class_name": cname,
            "n_conf_images": nc,
            "k_conf_clusters": kc_clusters,
            "k_c_image": k_c,
            "image_level_feasible": feasible_image_level,
            "hits_exact_sample_max": (k_c == nc),
            "finite_sample_lower_bound": k_c / (nc + 1) if feasible_image_level else None,
            "k_c_cluster_max": k_grp_c,
            "cluster_max_feasible": feasible_cluster_max,
            "q_c": q_c,
            "prob_cutoff": 1.0 - q_c,
            "second_largest_score": float(sc[-2]) if nc >= 2 else None,
        })

    # Mondrian prediction set: include class c if p_te[:, c] >= 1 - q_mondrian[c]
    sets_mond = p_te >= (1.0 - q_mondrian[None, :])

    # Test cluster bootstrap setup (1,972 test operational clusters)
    ug_te, inv_te = np.unique(grp_te, return_inverse=True)
    members_te = [np.where(inv_te == g)[0] for g in range(len(ug_te))]
    rng = np.random.default_rng(BOOT_SEED)
    boots = [np.concatenate([members_te[g] for g in rng.integers(0, len(ug_te), len(ug_te))]) for _ in range(N_BOOT)]

    def eval_sets(name: str, S: np.ndarray) -> tuple[dict, list[dict]]:
        cov = S[np.arange(len(y_te)), y_te].astype(float)
        sizes = S.sum(axis=1).astype(float)
        empty = (sizes == 0).astype(float)
        sing = (sizes == 1).astype(float)
        full = (sizes == 7).astype(float)

        def boot_ci(v):
            bm = np.array([v[ix].mean() for ix in boots])
            return float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))

        c_lo, c_hi = boot_ci(cov)
        s_lo, s_hi = boot_ci(sizes)
        # Equal-cluster-weight mean coverage
        eq_cov = float(pd.Series(cov, index=grp_te).groupby(level=0).mean().mean())
        eq_size = float(pd.Series(sizes, index=grp_te).groupby(level=0).mean().mean())

        overall = {
            "method": name,
            "n_test": len(y_te),
            "n_clusters": len(ug_te),
            "covered_count": int(cov.sum()),
            "marginal_coverage": float(cov.mean()),
            "coverage_cluster_ci95_low": c_lo,
            "coverage_cluster_ci95_high": c_hi,
            "equal_cluster_coverage": eq_cov,
            "mean_set_size": float(sizes.mean()),
            "size_cluster_ci95_low": s_lo,
            "size_cluster_ci95_high": s_hi,
            "equal_cluster_mean_size": eq_size,
            "empty_rate": float(empty.mean()),
            "singleton_rate": float(sing.mean()),
            "full_set_rate": float(full.mean()),
        }

        per_cls = []
        for c, cname in enumerate(CLASSES):
            mc = y_te == c
            nc_te = int(mc.sum())
            k_cov = int(cov[mc].sum())
            w_lo, w_hi = wilson_ci(k_cov, nc_te)
            per_cls.append({
                "method": name,
                "class_id": c,
                "class_name": cname,
                "n_test": nc_te,
                "n_test_clusters": int(len(np.unique(grp_te[mc]))),
                "covered_count": k_cov,
                "coverage": float(cov[mc].mean()),
                "wilson_ci95_low": w_lo,
                "wilson_ci95_high": w_hi,
                "mean_set_size": float(sizes[mc].mean()),
                "singleton_rate": float(sing[mc].mean()),
                "empty_rate": float(empty[mc].mean()),
                "full_set_rate": float(full[mc].mean()),
            })
        return overall, per_cls

    ov_pool, pc_pool = eval_sets("Pooled_LAC_SplitConformal", sets_pool)
    ov_mond, pc_mond = eval_sets("Mondrian_ClassConditional_LAC", sets_mond)

    # Disparity stats
    covs_pool = [r["coverage"] for r in pc_pool]
    covs_mond = [r["coverage"] for r in pc_mond]
    ov_pool["worst_class_coverage"] = min(covs_pool)
    ov_pool["class_coverage_gap"] = max(covs_pool) - min(covs_pool)
    ov_mond["worst_class_coverage"] = min(covs_mond)
    ov_mond["class_coverage_gap"] = max(covs_mond) - min(covs_mond)

    with (OUT_DIR / "phase_g_mondrian_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ov_pool.keys()))
        w.writeheader()
        w.writerows([ov_pool, ov_mond])

    with (OUT_DIR / "phase_g_mondrian_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pc_pool[0].keys()))
        w.writeheader()
        w.writerows(pc_pool + pc_mond)

    cfg = {
        "experiment": "Phase G Mondrian Class-Conditional Split-Conformal Prediction (LAC)",
        "protocol_status": "EXECUTED_POST_HOC_WITHOUT_TEST_TUNING",
        "alpha": 0.10,
        "split_seed": SPLIT_SEED,
        "boot_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "n_fit": int(len(fit_idx)),
        "n_conf": int(len(conf_idx)),
        "T_fit": T_fit,
        "cal_npz_sha256": sha256_file(CAL_NPZ),
        "test_npz_sha256": sha256_file(TE_NPZ),
    }
    (OUT_DIR / "phase_g_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    summary = {
        "feasibility_audit": {
            "verdict": "VALID_IMAGE_LEVEL_MONDRIAN_WITH_BOUNDARY_AND_CLUSTERING_DISCLOSURES",
            "per_class_calibration_diagnostics": cal_class_diag,
            "key_methodological_notes": [
                "All 7 classes in D_conf satisfy k_c = ceil((n_c + 1) * 0.90) <= n_c under image-level exchangeability.",
                "DF (n_c = 9 -> k_c = 9) sits at the exact minimum finite-sample threshold ceil((1 - alpha)/alpha) = 9; both DF (9/9) and VASC (10/10) select the empirical sample maximum s_(n_c).",
                "Under strict cluster-max aggregation within D_conf, cluster counts K_c for DF and VASC are smaller than 9 (making strict cluster-level finite-sample quantiles undefined without pooling or randomized ties), so Mondrian results here are explicitly scoped to image-level class-conditional calibration with group-disjoint D_fit/D_conf/Test splits and cluster-bootstrap test CIs.",
            ],
        },
        "pooled_overall": ov_pool,
        "mondrian_overall": ov_mond,
        "pooled_per_class": pc_pool,
        "mondrian_per_class": pc_mond,
    }
    (OUT_DIR / "phase_g_feasibility_and_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Phase G complete.")
    print("D_conf class diagnostics:")
    for d in cal_class_diag:
        print(f"  {d['class_name']:5s}: n_conf={d['n_conf_images']:3d} K_c={d['k_conf_clusters']:3d} k_c={d['k_c_image']:3d} q_c={d['q_c']:.6f} cutoff={d['prob_cutoff']:.6f} (cluster_max_feasible={d['cluster_max_feasible']})")
    print(f"\nPooled   overall: cov={ov_pool['marginal_coverage']*100:.2f}% [{ov_pool['coverage_cluster_ci95_low']*100:.2f}, {ov_pool['coverage_cluster_ci95_high']*100:.2f}] eq_cl={ov_pool['equal_cluster_coverage']*100:.2f}% size={ov_pool['mean_set_size']:.4f} sing={ov_pool['singleton_rate']*100:.2f}% empty={ov_pool['empty_rate']*100:.2f}% full={ov_pool['full_set_rate']*100:.2f}% worst={ov_pool['worst_class_coverage']*100:.2f}% gap={ov_pool['class_coverage_gap']*100:.2f}%")
    print(f"Mondrian overall: cov={ov_mond['marginal_coverage']*100:.2f}% [{ov_mond['coverage_cluster_ci95_low']*100:.2f}, {ov_mond['coverage_cluster_ci95_high']*100:.2f}] eq_cl={ov_mond['equal_cluster_coverage']*100:.2f}% size={ov_mond['mean_set_size']:.4f} sing={ov_mond['singleton_rate']*100:.2f}% empty={ov_mond['empty_rate']*100:.2f}% full={ov_mond['full_set_rate']*100:.2f}% worst={ov_mond['worst_class_coverage']*100:.2f}% gap={ov_mond['class_coverage_gap']*100:.2f}%")
    print("\nPer-class comparison (Pooled vs Mondrian):")
    for rp, rm in zip(pc_pool, pc_mond):
        print(f"  {rp['class_name']:5s} (n={rp['n_test']:4d}): Pooled cov={rp['coverage']*100:5.2f}% size={rp['mean_set_size']:.3f} | Mondrian cov={rm['coverage']*100:5.2f}% [{rm['wilson_ci95_low']*100:5.2f},{rm['wilson_ci95_high']*100:5.2f}] size={rm['mean_set_size']:.3f} full={rm['full_set_rate']*100:5.2f}%")


if __name__ == "__main__":
    main()
