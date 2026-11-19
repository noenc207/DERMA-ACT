"""Phase F: Controlled Post-Hoc Experiment 1 — Restricted-Action Learned Policy ({A1, A3, A4, A5, A6}).

Strictly read-only over frozen artifacts (per_action_test.npz, per_action_dev.npz, policy.pt,
lambda_selection.json, stopping_threshold.json). Zero CNN inference, zero policy retraining,
zero test-set tuning.

Outputs saved to research/derma_act/p8_1_final_submission_audit/phase_f_restricted_policy/.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score

import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modules.active_perception.policy import MultiHeadActionPolicy

import argparse

_ap = argparse.ArgumentParser(description="Phase F: Post-hoc candidate-action masking ablation on frozen V1 policy.")
_ap.add_argument("--root", type=Path, default=REPO_ROOT, help="Data root holding Tier B frozen predictions and metadata")
_ap.add_argument("--out-dir", type=Path, default=None, help="Output directory")
_pa, _ = _ap.parse_known_args()

ROOT = _pa.root.resolve()
OUT_DIR = (_pa.out_dir or (Path(__file__).resolve().parent)).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

TE_NPZ = ROOT / "research/derma_act/runs/derma_act_v0/predictions/per_action_test.npz"
DEV_NPZ = ROOT / "research/derma_act/evidence/v1/per_action_dev.npz"
TRACE_CSV = ROOT / "research/derma_act/evidence/v1/test_policy_trace.csv"
POL_PT = ROOT / "research/derma_act/runs/derma_act_v1/policy/policy.pt"
LAM_JSON = ROOT / "research/derma_act/runs/derma_act_v1/policy/lambda_selection.json"
TAU_JSON = ROOT / "research/derma_act/runs/derma_act_v1/policy/stopping_threshold.json"
META_CSV = ROOT / "data/isic2019/raw/ISIC_2019_Training_Metadata.csv"
P54_JSON = ROOT / "research/derma_act/p5_4_independent_policy_validation/P5_4_BENCHMARK_ANALYSIS.json"

ALLOWED_1IDX = [1, 3, 4, 5, 6]       # A1, A3, A4, A5, A6
ALLOWED_0IDX = [a - 1 for a in ALLOWED_1IDX]
SEQ_VAL = [6, 5, 3, 1, 4]
BOOT_SEED = 20261112
N_BOOT = 2000
EPS = 1e-12


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(1024 * 1024), b""):
            h.update(ch)
    return h.hexdigest()


def H(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(p + EPS), axis=-1)


def top2_margin(p: np.ndarray) -> np.ndarray:
    ps = np.sort(p, axis=-1)
    return ps[..., -1] - ps[..., -2]


def main() -> None:
    for req in (TE_NPZ, DEV_NPZ, TRACE_CSV, POL_PT, META_CSV):
        if not req.exists():
            raise FileNotFoundError(f'Missing Tier B artifact: {req}')
    torch.set_num_threads(1)
    sd = torch.load(POL_PT, map_location="cpu", weights_only=True)
    model = MultiHeadActionPolicy(state_dim=28, action_dim=9, hidden_dim=64)
    model.load_state_dict(sd)
    model.eval()

    lam = json.loads(LAM_JSON.read_text(encoding="utf-8"))["best_lambda"]
    lq, lc, li, lk = float(lam["lambda_Q"]), float(lam["lambda_C"]), float(lam["lambda_I"]), float(lam["lambda_K"])
    tau = float(json.loads(TAU_JSON.read_text(encoding="utf-8"))["best_tau"])

    # Feasibility check 1: input weight norms on mask bits
    w0 = sd["shared.0.weight"].numpy()
    mask_l2 = [float(np.linalg.norm(w0[:, 19 + i])) for i in range(9)]

    # Load frozen test & dev arrays
    te = np.load(TE_NPZ, allow_pickle=True)
    P_te = te["probabilities"].astype(np.float64)
    y_te = te["true_label"].astype(int)
    ids_te = te["image_id"].astype(str)
    N_test = len(y_te)

    dev = np.load(DEV_NPZ, allow_pickle=True)
    P_cal = dev["probabilities"][dev["split"] == "cal"].astype(np.float64)
    P_val = dev["probabilities"][dev["split"] == "val"].astype(np.float64)
    y_cal = dev["true_label"][dev["split"] == "cal"].astype(int)

    # Check Step 1 OOD sensitivity on CAL (without touching test for protocol choice)
    p0_cal = P_cal[:, 0, :].astype(np.float32)
    s_intact_cal = np.concatenate(
        [p0_cal, H(p0_cal)[:, None], p0_cal.max(1)[:, None], top2_margin(p0_cal)[:, None],
         np.zeros((len(p0_cal), 9), dtype=np.float32), np.ones((len(p0_cal), 9), dtype=np.float32)],
        axis=1,
    )
    mask_ood_cal = np.ones((len(p0_cal), 9), dtype=np.float32)
    for forbid in [1, 6, 7, 8]:
        mask_ood_cal[:, forbid] = 0.0
    s_ood_cal = s_intact_cal.copy()
    s_ood_cal[:, 19:28] = mask_ood_cal

    with torch.no_grad():
        def step1_diag(s_arr):
            u_mat = np.full((len(s_arr), len(ALLOWED_0IDX)), -np.inf, dtype=np.float64)
            for j, a0 in enumerate(ALLOWED_0IDX):
                oh = torch.zeros((len(s_arr), 9), dtype=torch.float32)
                oh[:, a0] = 1.0
                dq, dc, il = model(torch.from_numpy(s_arr), oh)
                ip = torch.sigmoid(il)
                u_mat[:, j] = (lq * dq + lc * dc - li * ip - lk).numpy()
            return float(u_mat.max(1).mean()), float((u_mat.max(1) < tau).mean())

        cal_mean_u_intact, cal_stop_intact = step1_diag(s_intact_cal)
        cal_mean_u_ood, cal_stop_ood = step1_diag(s_ood_cal)

    # Execute candidate-selection-restricted rollout on Test (frozen policy.pt, lambda_V1, tau_V1)
    def rollout_restricted(P: np.ndarray, ids: np.ndarray):
        N = len(P)
        p_t = P[:, 0, :].copy()
        history = np.zeros((N, 9), dtype=np.float32)
        state_mask = np.ones((N, 9), dtype=np.float32)  # maintains training invariant history + mask == 1
        stopped = np.zeros(N, dtype=bool)
        traj = np.zeros((N, 6, 7), dtype=np.float64)
        traj[:, 0, :] = p_t
        acqs = np.zeros(N, dtype=int)
        action_seq = [[] for _ in range(N)]
        trace_rows = []

        for b in range(1, 6):
            active = np.where(~stopped)[0]
            if len(active) < N:
                traj[~stopped, b, :] = p_t[~stopped]  # already updated below; propagate stopped below
            # Wait: propagate stopped carefully
            if len(active) == 0:
                traj[:, b, :] = p_t
                continue

            p_act = p_t[active].astype(np.float32)
            s_act = np.concatenate(
                [p_act, H(p_act)[:, None], p_act.max(1)[:, None], top2_margin(p_act)[:, None],
                 history[active], state_mask[active]],
                axis=1,
            ).astype(np.float32)
            s_t = torch.from_numpy(s_act)

            u_all = np.full((len(active), 9), -np.inf, dtype=np.float64)
            dq_all = np.zeros((len(active), 9), dtype=np.float64)
            dc_all = np.zeros((len(active), 9), dtype=np.float64)
            ip_all = np.zeros((len(active), 9), dtype=np.float64)
            with torch.no_grad():
                for a0 in ALLOWED_0IDX:
                    oh = torch.zeros((len(active), 9), dtype=torch.float32)
                    oh[:, a0] = 1.0
                    dq, dc, il = model(s_t, oh)
                    ip = torch.sigmoid(il)
                    u = (lq * dq + lc * dc - li * ip - lk).numpy()
                    avail = state_mask[active, a0] == 1.0
                    u_all[avail, a0] = u[avail]
                    dq_all[avail, a0] = dq.numpy()[avail]
                    dc_all[avail, a0] = dc.numpy()[avail]
                    ip_all[avail, a0] = ip.numpy()[avail]

            best_u = u_all.max(axis=1)
            best_a0 = u_all.argmax(axis=1)
            should_stop = best_u < tau

            for idx, i in enumerate(active):
                if should_stop[idx]:
                    stopped[i] = True
                    traj[i, b, :] = p_t[i]
                    trace_rows.append({
                        "image_id": ids[i], "step": b, "selected_action": "STOP",
                        "best_utility": float(best_u[idx]), "predicted_class": int(p_t[i].argmax()),
                    })
                else:
                    a0 = int(best_a0[idx])
                    a1 = a0 + 1
                    p_t[i] = (p_t[i] * b + P[i, a1]) / (b + 1)
                    history[i, a0] = 1.0
                    state_mask[i, a0] = 0.0
                    acqs[i] += 1
                    action_seq[i].append(a1)
                    traj[i, b, :] = p_t[i]
                    trace_rows.append({
                        "image_id": ids[i], "step": b, "selected_action": f"A{a1}",
                        "best_utility": float(best_u[idx]), "predicted_class": int(p_t[i].argmax()),
                    })
            # propagate already stopped before step b
            prev_stopped = np.ones(N, dtype=bool)
            prev_stopped[active] = False
            traj[prev_stopped, b, :] = p_t[prev_stopped]

        return traj, acqs, action_seq, trace_rows

    r_traj, r_acqs, r_seqs, r_traces = rollout_restricted(P_te, ids_te)
    pd.DataFrame(r_traces).to_csv(OUT_DIR / "phase_f_restricted_trace.csv", index=False)

    # Reconstruct canonical comparators from P5.4 logic to verify exact paired comparison
    p54 = json.loads(P54_JSON.read_text(encoding="utf-8"))
    cal_thrs = [float(p54["cal_frozen_thresholds"][f"B_{b}"]) for b in range(1, 6)]

    def rollout_conf_stop(P, c_thr, max_b):
        p = P[:, 0].copy()
        stopped = np.zeros(len(P), dtype=bool)
        cost = np.zeros(len(P), dtype=float)
        for b in range(1, max_b + 1):
            stop_now = (~stopped) & (p.max(axis=1) >= c_thr)
            stopped |= stop_now
            active = ~stopped
            a = SEQ_VAL[b - 1]
            p[active] = (p[active] * b + P[active, a]) / (b + 1)
            cost[active] += 1.0
        return p, cost, stopped

    # Learned V1 full from trace
    tr = pd.read_csv(TRACE_CSV)
    v1_traj = np.zeros((N_test, 6, 7), dtype=np.float64)
    v1_acqs = np.zeros(N_test, dtype=int)
    id2idx = {iid: k for k, iid in enumerate(ids_te)}
    import ast
    for iid, g in tr.groupby("image_id"):
        k = id2idx[iid]
        g = g.sort_values("step")
        steps = {int(r["step"]): np.array(ast.literal_eval(r["posterior"]), dtype=np.float64) for _, r in g.iterrows()}
        last_p = steps[0]
        v1_traj[k, 0] = last_p
        non_stop = g[(g["step"] >= 1) & (g["selected_action"] != "STOP")]["selected_action"].tolist()
        v1_acqs[k] = len(non_stop)
        for b in range(1, 6):
            if b in steps and len(non_stop) >= b:
                # posterior after b acquisitions is stored at step b+1 if stopped at b+1/exhausted, reconstruct exact
                pass
    # Exact posterior reconstruction from P_te and non_stop sequence (matches P5.4)
    for iid, g in tr.groupby("image_id"):
        k = id2idx[iid]
        g = g.sort_values("step")
        non_stop = g[(g["step"] >= 1) & (g["selected_action"] != "STOP")]["selected_action"].tolist()
        p = P_te[k, 0].copy()
        v1_traj[k, 0] = p
        for b in range(1, 6):
            if b <= len(non_stop):
                a = int(non_stop[b - 1][1:])
                p = (p * b + P_te[k, a]) / (b + 1)
            v1_traj[k, b] = p

    # Unstopped val-ranked
    traj_unstop = np.zeros((N_test, 6, 7), dtype=np.float64)
    traj_unstop[:, 0] = P_te[:, 0]
    vp = P_te[:, 0].copy()
    for b, a in enumerate(SEQ_VAL, 1):
        vp = (vp * b + P_te[:, a]) / (b + 1)
        traj_unstop[:, b] = vp

    # Clusters & bootstrap (exact P5.4 spec)
    meta = pd.read_csv(META_CSV).set_index("image")
    grp = np.array([meta.lesion_id.get(i) if isinstance(meta.lesion_id.get(i), str) else f"IMG::{i}" for i in ids_te])
    ug, inv = np.unique(grp, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(ug))]
    rng = np.random.default_rng(BOOT_SEED)
    boots = [np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))]) for _ in range(N_BOOT)]

    def cluster_ci(vals: np.ndarray):
        b_means = np.array([vals[ix].mean() for ix in boots])
        return [float(np.percentile(b_means, 2.5)), float(np.percentile(b_means, 97.5))]

    def paired_stat(pred_a, pred_b, comp_name, b_val):
        err_a = (pred_a != y_te).astype(int)
        err_b = (pred_b != y_te).astype(int)
        diff = err_a - err_b
        ci = cluster_ci(diff)
        a1_b0 = int(((err_a == 0) & (err_b == 1)).sum())
        a0_b1 = int(((err_a == 1) & (err_b == 0)).sum())
        n_disc = a1_b0 + a0_b1
        p_val = float(binomtest(a0_b1, n_disc, 0.5).pvalue) if n_disc > 0 else 1.0
        return {
            "comparison": comp_name,
            "budget": b_val,
            "risk_diff": float(diff.mean()),
            "ci95_low": ci[0],
            "ci95_high": ci[1],
            "a_correct_b_wrong": a1_b0,
            "a_wrong_b_correct": a0_b1,
            "mcnemar_p": p_val,
        }

    budget_rows = []
    paired_rows = []
    pred_a0 = P_te[:, 0].argmax(1)
    for b in range(1, 6):
        pr_res = r_traj[:, b].argmax(1)
        c_res = np.minimum(b, r_acqs)
        st_res = r_acqs < b
        err_res = (pr_res != y_te).astype(float)
        res_ci = cluster_ci(err_res)

        pr_v1 = v1_traj[:, b].argmax(1)
        p_cs, c_cs, st_cs = rollout_conf_stop(P_te, cal_thrs[b - 1], b)
        pr_cs = p_cs.argmax(1)
        pr_un = traj_unstop[:, b].argmax(1)

        budget_rows.append({
            "budget": b,
            "accuracy": float((pr_res == y_te).mean()),
            "risk": float(err_res.mean()),
            "risk_ci95_low": res_ci[0],
            "risk_ci95_high": res_ci[1],
            "balanced_accuracy": float(balanced_accuracy_score(y_te, pr_res)),
            "mean_cost": float(c_res.mean()),
            "early_stop_freq": float(st_res.mean()),
            "cal_stop_risk": float((pr_cs != y_te).mean()),
            "cal_stop_cost": float(c_cs.mean()),
            "learned_v1_9act_risk": float((pr_v1 != y_te).mean()),
            "learned_v1_9act_cost": float(np.minimum(b, v1_acqs).mean()),
        })

        paired_rows.append(paired_stat(pr_res, pr_cs, "Learned_V1_Spatial_vs_CAL_Stop", b))
        paired_rows.append(paired_stat(pr_res, pr_un, "Learned_V1_Spatial_vs_Unstopped_Spatial", b))
        paired_rows.append(paired_stat(pr_res, pr_v1, "Learned_V1_Spatial_vs_Learned_V1_9Action", b))
        paired_rows.append(paired_stat(pr_res, pred_a0, "Learned_V1_Spatial_vs_Global_A0", b))

    # Holm-Bonferroni over all 20 comparisons in Phase F (4 comparators x 5 budgets)
    m = len(paired_rows)
    order = np.argsort([r["mcnemar_p"] for r in paired_rows])
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, paired_rows[idx]["mcnemar_p"] * (m - rank))
        running_max = max(running_max, adj)
        paired_rows[idx]["holm_p_adj_m20"] = float(running_max)

    # Action usage across all executed acquisitions
    all_acts = [f"A{a}" for seq in r_seqs for a in seq]
    act_counts = dict(sorted(Counter(all_acts).items()))
    step1_acts = dict(sorted(Counter(f"A{seq[0]}" if seq else "STOP_B0" for seq in r_seqs).items()))
    # Check per-class accuracy at B=5
    class_names = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    pr_res_b5 = r_traj[:, 5].argmax(1)
    pr_v1_b5 = v1_traj[:, 5].argmax(1)
    p_cs_b5, _, _ = rollout_conf_stop(P_te, cal_thrs[4], 5)
    pr_cs_b5 = p_cs_b5.argmax(1)
    per_class_b5 = {}
    for cid, cname in enumerate(class_names):
        m_c = y_te == cid
        per_class_b5[cname] = {
            "n": int(m_c.sum()),
            "spatial_policy_sens": float((pr_res_b5[m_c] == cid).mean()),
            "v1_9act_sens": float((pr_v1_b5[m_c] == cid).mean()),
            "cal_stop_sens": float((pr_cs_b5[m_c] == cid).mean()),
        }

    with (OUT_DIR / "phase_f_comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(budget_rows[0].keys()))
        w.writeheader()
        w.writerows(budget_rows)

    with (OUT_DIR / "phase_f_paired_tests.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        w.writeheader()
        w.writerows(paired_rows)

    cfg = {
        "experiment": "Phase F Controlled Action-Restricted Learned Policy ({A1, A3, A4, A5, A6})",
        "protocol_status": "EXECUTED_POST_HOC_CANDIDATE_RESTRICTION",
        "allowed_actions": [f"A{a}" for a in ALLOWED_1IDX],
        "frozen_weights_sha256": sha256_file(POL_PT),
        "frozen_test_npz_sha256": sha256_file(TE_NPZ),
        "frozen_lambda": lam,
        "frozen_tau": tau,
        "boot_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "n_test": N_test,
        "n_clusters": len(ug),
    }
    (OUT_DIR / "phase_f_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    summary = {
        "feasibility_audit": {
            "verdict": "VALID_AS_POST_HOC_INFERENCE_RESTRICTION_WITH_CANDIDATE_MASKING",
            "why_ood_state_mask_rejected": (
                "Zeroing forbidden action bits (A2,A7,A8,A9) directly in the 28-dim MLP state vector violates the "
                "training invariant history[k]+mask[k]==1 (build_v1_states.py), feeding off-manifold orthants to "
                "non-trivial input weights. On CAL Step 1, OOD mask zeroing shifted mean predicted utility from "
                f"{cal_mean_u_intact:.4f} to {cal_mean_u_ood:.4f} and collapsed Step-1 stopping from "
                f"{cal_stop_intact*100:.2f}% to {cal_stop_ood*100:.2f}%."
            ),
            "mask_input_col_l2_norms_A1_to_A9": mask_l2,
            "what_is_controlled": "Both Learned_V1_Spatial and Val_Ranked_ConfStop/Unstopped draw exclusively from {A1,A3,A4,A5,A6} (same frozen classifier predictions).",
            "what_remains_uncontrolled": "policy.pt, lambda, and tau were trained/tuned under the 9-action regime, not retrained from scratch on 5 spatial actions.",
        },
        "budget_trajectory": budget_rows,
        "paired_comparisons_holm_m20": paired_rows,
        "action_counts_executed": act_counts,
        "step1_action_distribution": step1_acts,
        "total_acquisitions_executed": int(r_acqs.sum()),
        "per_class_b5_sensitivity": per_class_b5,
    }
    (OUT_DIR / "phase_f_feasibility_and_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Phase F complete. B=1..5 summary:")
    for r in budget_rows:
        print(f"  B={r['budget']}: SpatialV1 acc={r['accuracy']*100:.2f}% risk={r['risk']*100:.2f}% cost={r['mean_cost']:.3f} stop={r['early_stop_freq']*100:.2f}% | CAL-Stop risk={r['cal_stop_risk']*100:.2f}% cost={r['cal_stop_cost']:.3f} | 9-act V1 risk={r['learned_v1_9act_risk']*100:.2f}%")
    print("Paired B=5 comparisons:")
    for p in paired_rows:
        if p["budget"] == 5:
            print(f"  {p['comparison']}: dRisk={p['risk_diff']*100:+.2f}% CI=[{p['ci95_low']*100:+.2f}, {p['ci95_high']*100:+.2f}] disc={p['a_correct_b_wrong']}/{p['a_wrong_b_correct']} p={p['mcnemar_p']:.4g} holm={p['holm_p_adj_m20']:.4g}")
    print("Per-class B=5 sens:", json.dumps(per_class_b5, indent=2))
    print("Action counts:", act_counts)


if __name__ == "__main__":
    main()
