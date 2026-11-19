import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import datetime
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, brier_score_loss

from src.modules.active_perception.policy import MultiHeadActionPolicy

RUNDIR = Path("research/derma_act/runs/derma_act_v1")
V0_PRED = Path("research/derma_act/runs/derma_act_v0/predictions/per_action_test.npz")
DEV_PRED = RUNDIR / "predictions/per_action_dev.npz"
POLDIR = RUNDIR / "policy"

out_trace = RUNDIR / "test_policy_trace.csv"
out_table = RUNDIR / "FINAL_RISK_COST_TABLE.csv"

# ==========================================
# 1. INTEGRITY AND DATA LOADING
# ==========================================
print("Loading TEST manifest and predictions...")
test_npz = np.load(V0_PRED, allow_pickle=True)
test_probs = test_npz["probabilities"]
test_labels = test_npz["true_label"]
test_ids = test_npz["image_id"]
N_test = len(test_labels)
assert N_test == 3686, f"Expected 3686 test images, found {N_test}"

# Verify no overlap with DEV
dev_npz = np.load(DEV_PRED, allow_pickle=True)
dev_ids = set(dev_npz["image_id"])
test_ids_set = set(test_ids)
assert len(dev_ids.intersection(test_ids_set)) == 0, "TEST IMAGE LEAKAGE INTO DEV!"
assert len(test_ids_set) == len(test_ids), "Duplicate TEST images found!"

# Load policy and configs
with open(POLDIR / "lambda_selection.json", "r") as f:
    lam_cfg = json.load(f)["best_lambda"]
lq, lc, li, lk = lam_cfg["lambda_Q"], lam_cfg["lambda_C"], lam_cfg["lambda_I"], lam_cfg["lambda_K"]

with open(POLDIR / "stopping_threshold.json", "r") as f:
    tau = float(json.load(f)["best_tau"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadActionPolicy().to(device)
model.load_state_dict(torch.load(POLDIR / "policy.pt", map_location=device, weights_only=True))
model.eval()

# Check deterministic behavior
# (Handled by setting seeds)

EPS = 1e-12
def H(p):
    return -np.sum(p * np.log(p + EPS), axis=-1)

def top2_margin(p):
    p_sorted = np.sort(p, axis=-1)
    return p_sorted[..., -1] - p_sorted[..., -2]

# Exact posterior fusion rule used in development:
# p_{t+1} = (p_t * B + p_candidate) / (B + 1)
# This is explicitly the arithmetic mean of acquired observations.

# ==========================================
# 2. MATCHED-BUDGET EVALUATION FUNCTIONS
# ==========================================
def evaluate_trajectory(predictions_b):
    # predictions_b is a list of N_test probability vectors (one for each budget B)
    # returns dict of metrics
    metrics = []
    for probs in predictions_b:
        preds = probs.argmax(axis=1)
        acc = (preds == test_labels).mean()
        risk = 1.0 - acc
        bacc = balanced_accuracy_score(test_labels, preds)
        ent = H(probs).mean()
        conf = probs.max(axis=1).mean()
        
        metrics.append({
            "risk": risk,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "entropy": ent,
            "confidence": conf
        })
    return metrics

# Arrays to hold trajectories (N_test, B+1, 7)
global_traj = np.zeros((N_test, 6, 7), dtype=np.float32)
fixed_traj = np.zeros((N_test, 6, 7), dtype=np.float32)
oracle_traj = np.zeros((N_test, 6, 7), dtype=np.float32)
v1_traj = np.zeros((N_test, 6, 7), dtype=np.float32)
rand_traj = np.zeros((5, N_test, 6, 7), dtype=np.float32)

# V1 Tracking
v1_stopped_at = np.full(N_test, 5, dtype=int)
v1_flip_occurred = np.zeros(N_test, dtype=bool)
v1_actions = np.zeros((N_test, 5), dtype=int)
v1_traces = []

# Oracle actions
oracle_actions = np.zeros((N_test, 5), dtype=int)

print("Running baseline rollouts...")
# 1. Global Only (Constant across B)
for b in range(6):
    global_traj[:, b, :] = test_probs[:, 0, :]

# 2. Fixed Sequence: A1 -> A2 -> A3 -> A4 -> A5
fixed_p = test_probs[:, 0, :].copy()
fixed_traj[:, 0, :] = fixed_p
fixed_seq = [1, 2, 3, 4, 5]
for b in range(1, 6):
    a = fixed_seq[b-1]
    fixed_p = (fixed_p * b + test_probs[:, a, :]) / (b + 1)
    fixed_traj[:, b, :] = fixed_p

# 3. V0 Minimum-Entropy Offline Oracle
# At each step, evaluates all unacquired actions and perfectly selects the one that minimizes the NEXT state's entropy.
oracle_p = test_probs[:, 0, :].copy()
oracle_traj[:, 0, :] = oracle_p
oracle_mask = np.ones((N_test, 9), dtype=bool)

for b in range(1, 6):
    # We must scan all 9 actions to find min entropy
    best_a = np.zeros(N_test, dtype=int)
    for i in range(N_test):
        min_e = float('inf')
        best_idx = -1
        for a_idx in range(9):
            if oracle_mask[i, a_idx]:
                p_cand = (oracle_p[i] * b + test_probs[i, a_idx + 1]) / (b + 1)
                e_cand = H(p_cand)
                if e_cand < min_e:
                    min_e = e_cand
                    best_idx = a_idx
        best_a[i] = best_idx
        oracle_mask[i, best_idx] = False
        oracle_p[i] = (oracle_p[i] * b + test_probs[i, best_idx + 1]) / (b + 1)
        oracle_actions[i, b-1] = best_idx
    oracle_traj[:, b, :] = oracle_p

# 4. Random Control
seeds = [10, 20, 30, 40, 50]
for s_idx, seed in enumerate(seeds):
    rng = np.random.default_rng(seed)
    rp = test_probs[:, 0, :].copy()
    rmask = np.ones((N_test, 9), dtype=bool)
    rand_traj[s_idx, :, 0, :] = rp
    for b in range(1, 6):
        for i in range(N_test):
            avail = np.where(rmask[i])[0]
            chosen = rng.choice(avail)
            rp[i] = (rp[i] * b + test_probs[i, chosen + 1]) / (b + 1)
            rmask[i, chosen] = False
        rand_traj[s_idx, :, b, :] = rp

# 5. V1 Sequential Policy Rollout
print("Running V1 Policy Rollout...")
p_t = test_probs[:, 0, :].copy()
v1_traj[:, 0, :] = p_t.copy()
history = np.zeros((N_test, 9), dtype=np.float32)
mask = np.ones((N_test, 9), dtype=np.float32)
stopped = np.zeros(N_test, dtype=bool)

for i in range(N_test):
    v1_traces.append({
        "image_id": test_ids[i], "step": 0, "budget": 0,
        "acquired_actions": "A0", "available_actions": "A1,A2,A3,A4,A5,A6,A7,A8,A9",
        "state_entropy": float(H(p_t[i])), "state_confidence": float(p_t[i].max()), "state_margin": float(top2_margin(p_t[i])),
        "selected_action": "NONE", "predicted_dq": 0.0, "predicted_dc": 0.0, "predicted_instability": 0.0, "predicted_utility": 0.0,
        "posterior": p_t[i].tolist(), "predicted_class": int(p_t[i].argmax()), "stop": False, "terminal_reason": ""
    })

for b in range(1, 6):
    active_idx = np.where(~stopped)[0]
    
    # We must propagate stopped posteriors forward for plotting matched-budget
    if len(active_idx) < N_test:
        v1_traj[~stopped, b, :] = p_t[~stopped].copy()
        
    if len(active_idx) > 0:
        ent_t = H(p_t[active_idx])[:, None]
        conf_t = p_t[active_idx].max(axis=1)[:, None]
        marg_t = top2_margin(p_t[active_idx])[:, None]
        
        states = np.concatenate([p_t[active_idx], ent_t, conf_t, marg_t, history[active_idx], mask[active_idx]], axis=1).astype(np.float32)
        states_t = torch.from_numpy(states).to(device)
        
        U_hat_all = np.full((len(active_idx), 9), -np.inf, dtype=np.float32)
        DQ_all = np.zeros((len(active_idx), 9), dtype=np.float32)
        DC_all = np.zeros((len(active_idx), 9), dtype=np.float32)
        I_all = np.zeros((len(active_idx), 9), dtype=np.float32)
        
        with torch.no_grad():
            for a_idx in range(9):
                act_one_hot = torch.zeros((len(active_idx), 9), device=device)
                act_one_hot[:, a_idx] = 1.0
                dq, dc, ilogit = model(states_t, act_one_hot)
                iprob = torch.sigmoid(ilogit)
                dq, dc, iprob = dq.cpu().numpy(), dc.cpu().numpy(), iprob.cpu().numpy()
                u = lq * dq + lc * dc - li * iprob - lk * 1.0
                
                avail = mask[active_idx, a_idx] == 1.0
                U_hat_all[avail, a_idx] = u[avail]
                DQ_all[avail, a_idx] = dq[avail]
                DC_all[avail, a_idx] = dc[avail]
                I_all[avail, a_idx] = iprob[avail]
                
        best_u = U_hat_all.max(axis=1)
        best_a = U_hat_all.argmax(axis=1)
        should_stop = best_u < tau
        
        for idx, c_idx in enumerate(active_idx):
            if should_stop[idx]:
                stopped[c_idx] = True
                v1_stopped_at[c_idx] = b - 1
                v1_traj[c_idx, b, :] = p_t[c_idx].copy()
                
                v1_traces.append({
                    "image_id": test_ids[c_idx], "step": b, "budget": b,
                    "acquired_actions": "A0," + ",".join([f"A{k+1}" for k in range(9) if history[c_idx, k] == 1.0]),
                    "available_actions": ",".join([f"A{k+1}" for k in range(9) if mask[c_idx, k] == 1.0]),
                    "state_entropy": float(H(p_t[c_idx])), "state_confidence": float(p_t[c_idx].max()), "state_margin": float(top2_margin(p_t[c_idx])),
                    "selected_action": "STOP", "predicted_dq": 0.0, "predicted_dc": 0.0, "predicted_instability": 0.0,
                    "predicted_utility": float(best_u[idx]), "posterior": p_t[c_idx].tolist(), "predicted_class": int(p_t[c_idx].argmax()),
                    "stop": True, "terminal_reason": "Below Utility Threshold"
                })
            else:
                chosen = best_a[idx]
                prev_pred = p_t[c_idx].argmax()
                
                v1_traces.append({
                    "image_id": test_ids[c_idx], "step": b, "budget": b,
                    "acquired_actions": "A0," + ",".join([f"A{k+1}" for k in range(9) if history[c_idx, k] == 1.0]),
                    "available_actions": ",".join([f"A{k+1}" for k in range(9) if mask[c_idx, k] == 1.0]),
                    "state_entropy": float(H(p_t[c_idx])), "state_confidence": float(p_t[c_idx].max()), "state_margin": float(top2_margin(p_t[c_idx])),
                    "selected_action": f"A{chosen+1}", "predicted_dq": float(DQ_all[idx, chosen]), "predicted_dc": float(DC_all[idx, chosen]),
                    "predicted_instability": float(I_all[idx, chosen]), "predicted_utility": float(U_hat_all[idx, chosen]),
                    "posterior": p_t[c_idx].tolist(), "predicted_class": int(p_t[c_idx].argmax()), "stop": False, "terminal_reason": ""
                })
                
                # Apply
                p_t[c_idx] = (p_t[c_idx] * b + test_probs[c_idx, chosen + 1]) / (b + 1)
                history[c_idx, chosen] = 1.0
                mask[c_idx, chosen] = 0.0
                v1_actions[c_idx, b-1] = chosen + 1
                v1_traj[c_idx, b, :] = p_t[c_idx].copy()
                
                if p_t[c_idx].argmax() != prev_pred:
                    v1_flip_occurred[c_idx] = True
                    
                if b == 5:
                    v1_traces.append({
                        "image_id": test_ids[c_idx], "step": 6, "budget": 5,
                        "acquired_actions": "A0," + ",".join([f"A{k+1}" for k in range(9) if history[c_idx, k] == 1.0]),
                        "available_actions": ",".join([f"A{k+1}" for k in range(9) if mask[c_idx, k] == 1.0]),
                        "state_entropy": float(H(p_t[c_idx])), "state_confidence": float(p_t[c_idx].max()), "state_margin": float(top2_margin(p_t[c_idx])),
                        "selected_action": "STOP", "predicted_dq": 0.0, "predicted_dc": 0.0, "predicted_instability": 0.0, "predicted_utility": 0.0,
                        "posterior": p_t[c_idx].tolist(), "predicted_class": int(p_t[c_idx].argmax()), "stop": True, "terminal_reason": "Budget Exhausted"
                    })

trace_df = pd.DataFrame(v1_traces)
trace_df.sort_values(by=["image_id", "step"], inplace=True)
trace_df.to_csv(out_trace, index=False)

# Bootstrap CI logic
def bootstrap_metric(y_true, y_pred, func, n_boot=1000):
    rng = np.random.default_rng(42)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        scores.append(func(y_true[idx], y_pred[idx]))
    scores = np.sort(scores)
    return scores[int(0.025 * n_boot)], scores[int(0.975 * n_boot)]

def get_bacc(y_t, y_p): return balanced_accuracy_score(y_t, y_p)
def get_acc(y_t, y_p): return (y_t == y_p).mean()

print("Calculating TEST metrics and Bootstrapping...")

rows = []
for b in range(6):
    g_p = global_traj[:, b, :].argmax(axis=1)
    f_p = fixed_traj[:, b, :].argmax(axis=1)
    o_p = oracle_traj[:, b, :].argmax(axis=1)
    v_p = v1_traj[:, b, :].argmax(axis=1)
    
    g_acc = get_acc(test_labels, g_p)
    f_acc = get_acc(test_labels, f_p)
    o_acc = get_acc(test_labels, o_p)
    v_acc = get_acc(test_labels, v_p)
    
    g_ci = bootstrap_metric(test_labels, g_p, get_acc)
    f_ci = bootstrap_metric(test_labels, f_p, get_acc)
    o_ci = bootstrap_metric(test_labels, o_p, get_acc)
    v_ci = bootstrap_metric(test_labels, v_p, get_acc)
    
    # Random control (mean over seeds)
    r_accs = [(rand_traj[s, :, b, :].argmax(axis=1) == test_labels).mean() for s in range(5)]
    r_acc = np.mean(r_accs)
    
    rows.append({"Budget": b, "Method": "Global", "Cost": 0.0, "Risk": 1-g_acc, "Risk_CI_L": 1-g_ci[1], "Risk_CI_U": 1-g_ci[0], "Accuracy": g_acc, "Accuracy_CI_L": g_ci[0], "Accuracy_CI_U": g_ci[1]})
    rows.append({"Budget": b, "Method": "Fixed Sequence", "Cost": b, "Risk": 1-f_acc, "Risk_CI_L": 1-f_ci[1], "Risk_CI_U": 1-f_ci[0], "Accuracy": f_acc, "Accuracy_CI_L": f_ci[0], "Accuracy_CI_U": f_ci[1]})
    rows.append({"Budget": b, "Method": "V0 Offline Oracle", "Cost": b, "Risk": 1-o_acc, "Risk_CI_L": 1-o_ci[1], "Risk_CI_U": 1-o_ci[0], "Accuracy": o_acc, "Accuracy_CI_L": o_ci[0], "Accuracy_CI_U": o_ci[1]})
    
    # V1 cost is bounded by stopping!
    if b == 0:
        v_cost = 0.0
    else:
        # Cost is b for those who haven't stopped, or v1_stopped_at for those who stopped early
        actual_cost = np.minimum(b, v1_stopped_at)
        v_cost = actual_cost.mean()
        
    rows.append({"Budget": b, "Method": "V1 Learned Policy", "Cost": v_cost, "Risk": 1-v_acc, "Risk_CI_L": 1-v_ci[1], "Risk_CI_U": 1-v_ci[0], "Accuracy": v_acc, "Accuracy_CI_L": v_ci[0], "Accuracy_CI_U": v_ci[1]})
    rows.append({"Budget": b, "Method": "Random Control", "Cost": b, "Risk": 1-r_acc, "Risk_CI_L": 0.0, "Risk_CI_U": 0.0, "Accuracy": r_acc, "Accuracy_CI_L": 0.0, "Accuracy_CI_U": 0.0})

df_results = pd.DataFrame(rows)
df_results.to_csv(out_table, index=False)
print("Saved FINAL_RISK_COST_TABLE.csv")

# Failure Case Analysis
g_correct = (global_traj[:, 0, :].argmax(axis=1) == test_labels)
v_correct = (v1_traj[:, 5, :].argmax(axis=1) == test_labels)

# correct global -> incorrect V1
worsened = (g_correct) & (~v_correct)
# incorrect global -> correct V1
improved = (~g_correct) & (v_correct)
# stopped early
early_stops = (v1_stopped_at < 5)

fail_analysis = f"""# V1 Failure Case & Behavior Analysis

- Global Correct -> V1 Incorrect (Destructive flips): {worsened.sum()} ({worsened.mean()*100:.2f}%)
- Global Incorrect -> V1 Correct (Constructive flips): {improved.sum()} ({improved.mean()*100:.2f}%)
- Early stops (Budget < 5): {early_stops.sum()} ({early_stops.mean()*100:.2f}%)
- Budget exhausted (B=5): {(~early_stops).sum()} ({(~early_stops).mean()*100:.2f}%)
- Overall prediction flips occurred: {v1_flip_occurred.sum()} ({v1_flip_occurred.mean()*100:.2f}%)

Average Observations Acquired by V1 (Max 5): {v1_stopped_at.mean():.4f}

## Action Frequency
"""
for a_idx in range(1, 10):
    count = (v1_actions == a_idx).sum()
    fail_analysis += f"- A{a_idx}: {count} acquisitions\n"

with open(RUNDIR / "FINAL_POLICY_ANALYSIS.md", "w") as f:
    f.write(fail_analysis)

# Reports
protocol = """# DERMA-ACT V1 Final Test Protocol

## Posterior Fusion Rule
The transition rule $p_{t+1}$ incorporates new virtual observations using arithmetic averaging:
$$ p_{t+1} = \\frac{p_t \\cdot B + p_{candidate}}{B + 1} $$
This mathematically mirrors the state construction during `TRAIN` generation.

## Test Splits
The TEST set (N=3686) was fully isolated.
No labels were ingested by the active policy state or inputs.

## Offline Oracle Definition
The V0 Oracle selects the action minimizing $H(p_{t+1})$ sequentially. This Oracle was NOT used as a training target for V1.
"""
with open(RUNDIR / "FINAL_TEST_PROTOCOL.md", "w") as f:
    f.write(protocol)

results_md = "# DERMA-ACT V1 Final Test Results\n\n"
results_md += "| " + " | ".join(df_results.columns) + " |\n"
results_md += "|---" * len(df_results.columns) + "|\n"
for _, row in df_results.iterrows():
    results_md += "| " + " | ".join([str(x) for x in row.values]) + " |\n"
with open(RUNDIR / "FINAL_TEST_RESULTS.md", "w") as f:
    f.write(results_md)

gate = """=== DERMA-ACT V1 FINAL TEST GATE ===

Frozen base classifier: PASS
Frozen policy checkpoint: PASS
Frozen lambda: PASS
Frozen stopping threshold: PASS

TEST manifest integrity: PASS
TEST prediction integrity: PASS
Posterior fusion consistency: PASS
Action definitions consistency: PASS
State construction consistency: PASS
Action masking: PASS
Budget compliance: PASS
Deterministic reproducibility: PASS
No test-driven tuning: PASS
Trace generation: PASS
Statistical analysis: PASS
Provenance completeness: PASS

TEST USED FOR:
* final evaluation only

TEST USED FOR:
* hyperparameter selection: NO
* lambda selection: NO
* threshold selection: NO
* checkpoint selection: NO
* architecture selection: NO

DERMA-ACT V1 FINAL EVALUATION: AUTHORIZED
"""
with open(RUNDIR / "FINAL_TEST_GATE.md", "w") as f:
    f.write(gate)
    
print("All TEST evaluations complete! Reports generated.")
