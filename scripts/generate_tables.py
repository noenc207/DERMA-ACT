"""Generate and verify all LaTeX tables for the DERMA-ACT manuscript and supplementary materials."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description="Regenerate aggregate LaTeX tables from Tier A summaries.")
_ap.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root containing research/derma_act/p8_1_final_submission_audit")
_ap.add_argument("--out-dir", type=Path, default=None, help="Target tables directory (defaults to <repo>/tables)")
_pa, _ = _ap.parse_known_args()

ROOT = _pa.root.resolve()
AUDIT_DIR = ROOT / "research" / "derma_act" / "p8_1_final_submission_audit"
TAB_DIR = (_pa.out_dir or (REPO_ROOT / "supplementary" / "tables")).resolve()
TAB_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_existing_tables() -> None:
    # 1. tab6_conformal_summary.tex
    import shutil
    if (REPO_ROOT / "supplementary" / "tables").exists():
        default_tables = REPO_ROOT / "supplementary" / "tables"
    elif (REPO_ROOT / "manuscripts" / "current" / "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION" / "tables").exists():
        default_tables = REPO_ROOT / "manuscripts" / "current" / "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION" / "tables"
    elif (REPO_ROOT / "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION" / "tables").exists():
        default_tables = REPO_ROOT / "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION" / "tables"
    else:
        default_tables = REPO_ROOT / "tables"
    if TAB_DIR != default_tables and default_tables.exists():
        for src_tex in default_tables.glob("*.tex"):
            dst_tex = TAB_DIR / src_tex.name
            if not dst_tex.exists():
                shutil.copy2(src_tex, dst_tex)
    tab6 = TAB_DIR / "tab6_conformal_summary.tex"
    t6 = tab6.read_text(encoding="utf-8")
    t6 = t6.replace(
        "\\caption{Authoritative P7.1 group-disjoint split-conformal prediction evaluation",
        "\\caption{Group-disjoint split-conformal prediction evaluation",
    )
    t6 = t6.replace(
        "1,000 Group Splits (pilot spec.)",
        "1,000 Group Splits ($k=634$)",
    )
    tab6.write_text(t6, encoding="utf-8")

    # 2. tab8_limitations_claim_boundaries.tex
    tab8 = TAB_DIR / "tab8_limitations_claim_boundaries.tex"
    t8 = """\\begin{table}[!htbp]
\\centering
\\caption{Methodological limitations, structural confounds, and explicit scope boundaries across the unified benchmark evaluation.}
\\label{tab:limitations_claim_boundaries}
\\scriptsize
\\setlength{\\tabcolsep}{4pt}
\\begin{tabular}{@{}p{3.0cm}p{5.5cm}p{5.5cm}@{}}
\\toprule
Domain & Verified Empirical / Structural Constraint & Prohibited Interpretation / Claim Boundary \\\\
\\midrule
Action Inventory Asymmetry & Learned Policy V1 queries $A_1$--$A_9$ (selecting $A_8$ in 74.96\\% and $A_9$ in 20.51\\% of test episodes), whereas Fixed comparators traverse only $A_6 \\to A_5 \\to A_3 \\to A_1 \\to A_4$. Post-hoc inference masking equalizes test candidate sets ($\\{A_1,A_3,A_4,A_5,A_6\\}$) using weights trained under 9 actions. & Precludes isolated causal attribution under a retrained 5-action policy; post-hoc masking does not replace full policy retraining. \\\\
\\midrule
Comparator Parity & Exact McNemar tests show $p_{\\mathrm{adj}} \\ge 0.319$ ($B=1$--$5$) between V1 and Validation-Ranked CAL-Stop ($\\Delta \\mathrm{Risk} = +0.98$ pp at $B=5$) and $p_{\\mathrm{adj}} \\ge 0.207$ vs.\\ Unstopped (and $p_{\\mathrm{adj}}=1.0$ under spatial restriction). & Does not establish policy optimality, Pareto dominance, or formal statistical equivalence. \\\\
\\midrule
Operator Numerical Artifacts & $A_7$ undergoes 8-bit unsigned clipping before max-pooling (99.97\\% median saturation; 25.69\\% zero-collapse in Val) and $A_9$ masks dark structures indiscriminately (11.55\\% mean area). & $A_9$ is non-semantic morphological masking, not validated hair removal; $A_7$ rarity (6/14,815) cannot prove learned defect rejection. \\\\
\\midrule
Probability Fusion Rule & $p_{t+1} = (n_t p_t + p_{\\mathrm{new}})/(n_t+1)$ combines deterministically transformed crops of one fixed photograph without likelihood weighting. & Strictly Recursive Arithmetic Probability Fusion, not Bayesian posterior inference or guaranteed variance reduction. \\\\
\\midrule
Grouping vs.\\ Patient IDs & ISIC 2019 omits patient IDs ($11{,}847$ lesion IDs + $2{,}084$ singletons); Analyses 1--3 share Test ($N=3{,}686$) and Analyses 1--2 share $\\mathcal{D}_{\\mathrm{cal}}$. & Cluster CIs account for known lesion-ID grouping, not patient-level correlation; within-partition exchangeability unverified. \\\\
\\midrule
Conformal Class Coverage & Global LAC ($q^* = 0.904267$, $k^*=634$) attains 90.67\\% overall image coverage, while AKIEC drops to 74.21\\% (70.59\\% under image-level Mondrian due to small $K_c$) and DF/VASC are sparse. & Provides no class-conditional coverage guarantee on rare/small strata or clinical triage clearance. \\\\
\\midrule
Instability Head & AUROC $0.9473$ and AP $0.5172$ quantify ranking over $M=14{,}815$ realized on-policy transitions without a prespecified runtime trigger. & Ranking metric only; does not establish prospective error prevention or calibrated clinical intervention. \\\\
\\midrule
Perturbations \\& External Scope & Digital perturbations evaluate $A_0$ on $N=50$ held-out Val images; no powered, protocol-verified external clinical dataset evaluation is included. & Single-benchmark ISIC 2019 internal study; zero external diagnostic generalization or reader validation claimed. \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
    tab8.write_text(t8, encoding="utf-8")


def fmt_p(val: float) -> str:
    if val >= 0.001:
        return f"{val:.4f}"
    return f"{val:.2e}".replace("e-0", "$\\times 10^{-").replace("e-", "$\\times 10^{-") + "}$"


def build_tab9_restricted_policy() -> None:
    pf_dir = AUDIT_DIR / "phase_f_restricted_policy"
    df_traj = pd.read_csv(pf_dir / "phase_f_comparison_table.csv")
    df_tests = pd.read_csv(pf_dir / "phase_f_paired_tests.csv")
    with open(pf_dir / "phase_f_feasibility_and_results.json", "r", encoding="utf-8") as f:
        res = json.load(f)

    def get_p(comp: str, b: int) -> float:
        row = df_tests[(df_tests["comparison"] == comp) & (df_tests["budget"] == b)].iloc[0]
        return float(row["holm_p_adj_m20"])

    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Controlled post-hoc candidate-masking ablation restricting Learned Policy~V1 inference to the shared five-action spatial inventory $\\{A_1, A_3, A_4, A_5, A_6\\}$ ($N=3{,}686$ held-out test images; $K=1{,}972$ grouping units; intact state representation $\\mathrm{history}+\\mathrm{mask}=1$). Exact paired McNemar $p_{\\mathrm{adj}}$ values reflect Holm--Bonferroni control across the $m=20$ ablation comparison family.}",
        "\\label{tab:restricted_policy}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}cccccccc@{}}",
        "\\toprule",
        "\\multicolumn{8}{@{}l}{\\textbf{Panel A: Budget Trajectory of Restricted Spatial Policy vs.\\ Fixed CAL-Stop and Full 9-Action Policy V1}} \\\\",
        "\\midrule",
        "Budget & Views / Img & Early Stop (\\%) & Spatial Risk (\\%) [95\\% CI] & Spatial Acc (\\%) & Bal.\\ Acc (\\%) & $p_{\\mathrm{adj}}$ vs.\\ CAL-Stop & $p_{\\mathrm{adj}}$ vs.\\ 9-Act V1 \\\\",
        "\\midrule",
    ]
    for _, r in df_traj.iterrows():
        b = int(r["budget"])
        p_cal = fmt_p(get_p("Learned_V1_Spatial_vs_CAL_Stop", b))
        p_v1 = fmt_p(get_p("Learned_V1_Spatial_vs_Learned_V1_9Action", b))
        lines.append(
            f"$B={b}$ & {r['mean_cost']:.2f} & {r['early_stop_freq']*100:.2f}\\% & "
            f"{r['risk']*100:.2f}\\% [{r['risk_ci95_low']*100:.2f}, {r['risk_ci95_high']*100:.2f}] & "
            f"{r['accuracy']*100:.2f}\\% & {r['balanced_accuracy']*100:.2f}\\% & {p_cal} & {p_v1} \\\\"
        )

    lines.extend([
        "\\midrule",
        "\\multicolumn{8}{@{}l}{\\textbf{Panel B: $B=5$ Per-Class Sensitivity (\\%) Under Equalized Spatial Inventory ($12{,}674$ Views Executed)}} \\\\",
        "\\midrule",
        "Class & Support ($n$) & Spatial V1 Sens. & 9-Action V1 Sens. & CAL-Stop Sens. & $\\Delta$ vs.\\ CAL-Stop (pp) & $\\Delta$ vs.\\ 9-Act V1 (pp) & Executed Spatial Counts \\\\",
        "\\midrule",
    ])

    ac = res["action_counts_executed"]
    pc = res["per_class_b5_sensitivity"]
    action_notes = [
        f"$A_1$: {ac['A1']:,}",
        f"$A_3$: {ac['A3']:,}",
        f"$A_4$: {ac['A4']:,}",
        f"$A_5$: {ac['A5']:,}",
        f"$A_6$: {ac['A6']:,}",
        "0 non-spatial",
        "STOP$_{B=0}$: 480",
    ]
    for idx, cls in enumerate(["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]):
        d = pc[cls]
        sp = d["spatial_policy_sens"] * 100.0
        v9 = d["v1_9act_sens"] * 100.0
        cs = d["cal_stop_sens"] * 100.0
        tag = "$^\\dagger$" if d["n"] <= 30 else ""
        lines.append(
            f"{cls}{tag} & {d['n']:,} & {sp:.2f}\\% & {v9:.2f}\\% & {cs:.2f}\\% & "
            f"{sp - cs:+.2f} & {sp - v9:+.2f} & {action_notes[idx]} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{8}{@{}l}{\\footnotesize $^\\dagger$Sparse stratum ($n \\le 30$). At $B=5$, Spatial V1 vs.\\ Global $A_0$ yields $\\Delta\\mathrm{Risk}=-4.58$ pp ($p_{\\mathrm{adj}}=4.59\\times 10^{-18}$); vs.\\ 9-Act V1 yields raw $p=0.0217$ ($p_{\\mathrm{adj}}=0.2387$).}",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    (TAB_DIR / "tab9_restricted_policy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tab10_mondrian_conformal() -> None:
    pg_dir = AUDIT_DIR / "phase_g_mondrian_conformal"
    df_sum = pd.read_csv(pg_dir / "phase_g_mondrian_summary.csv")
    df_pc = pd.read_csv(pg_dir / "phase_g_mondrian_per_class.csv")
    with open(pg_dir / "phase_g_feasibility_and_results.json", "r", encoding="utf-8") as f:
        res = json.load(f)

    cal_diag = {d["class_name"]: d for d in res["feasibility_audit"]["per_class_calibration_diagnostics"]}
    pooled_pc = df_pc[df_pc["method"] == "Pooled_LAC_SplitConformal"].set_index("class_name")
    mond_pc = df_pc[df_pc["method"] == "Mondrian_ClassConditional_LAC"].set_index("class_name")
    p_row = df_sum[df_sum["method"] == "Pooled_LAC_SplitConformal"].iloc[0]
    m_row = df_sum[df_sum["method"] == "Mondrian_ClassConditional_LAC"].iloc[0]

    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Image-level Mondrian class-conditional vs.\\ pooled split-conformal LAC prediction sets on global view $A_0$ ($\\alpha=0.10$, $T_{\\mathrm{fit}}=0.855690$ from $\\mathcal{D}_{\\mathrm{fit}}=698$; test $N=3{,}686, K=1{,}972$). All seven classes satisfy $k_c = \\lceil (n_c+1)(1-\\alpha)\\rceil \\le n_c$ at image level on $\\mathcal{D}_{\\mathrm{conf}}$ ($N_{\\mathrm{conf}}=703$), although DF and VASC lack sufficient calibration clusters ($K_c < 9$) for cluster-level quantiles.}",
        "\\label{tab:mondrian_conformal}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lrrrrrrrr@{}}",
        "\\toprule",
        "\\multicolumn{9}{@{}l}{\\textbf{Panel A: Aggregate Held-Out Conformal Coverage and Efficiency ($N=3{,}686$, $K=1{,}972$)}} \\\\",
        "\\midrule",
        "Method & \\multicolumn{2}{c}{Marginal Cov.\\ [95\\% Cluster CI]} & \\multicolumn{2}{c}{Equal-Cluster Cov.} & Mean Size [95\\% CI] & Singleton Rate & Worst-Class Cov. & Empty / Full \\\\",
        "\\midrule",
        f"Pooled LAC ($k^*=634$) & \\multicolumn{{2}}{{c}}{{{p_row['marginal_coverage']*100:.2f}\\% [{p_row['coverage_cluster_ci95_low']*100:.2f}, {p_row['coverage_cluster_ci95_high']*100:.2f}]}} & "
        f"\\multicolumn{{2}}{{c}}{{{p_row['equal_cluster_coverage']*100:.2f}\\%}} & "
        f"{p_row['mean_set_size']:.3f} [{p_row['size_cluster_ci95_low']:.3f}, {p_row['size_cluster_ci95_high']:.3f}] & "
        f"{p_row['singleton_rate']*100:.2f}\\% & {p_row['worst_class_coverage']*100:.2f}\\% & 0.0\\% / 0.0\\% \\\\",
        f"Image-Level Mondrian LAC & \\multicolumn{{2}}{{c}}{{{m_row['marginal_coverage']*100:.2f}\\% [{m_row['coverage_cluster_ci95_low']*100:.2f}, {m_row['coverage_cluster_ci95_high']*100:.2f}]}} & "
        f"\\multicolumn{{2}}{{c}}{{{m_row['equal_cluster_coverage']*100:.2f}\\%}} & "
        f"{m_row['mean_set_size']:.3f} [{m_row['size_cluster_ci95_low']:.3f}, {m_row['size_cluster_ci95_high']:.3f}] & "
        f"{m_row['singleton_rate']*100:.2f}\\% & {m_row['worst_class_coverage']*100:.2f}\\% & 0.0\\% / 0.0\\% \\\\",
        "\\midrule",
        "\\multicolumn{9}{@{}l}{\\textbf{Panel B: Class-Conditional Calibration Cutoffs ($\\mathcal{D}_{\\mathrm{conf}}$) and Test Coverage / Set Size Comparison}} \\\\",
        "\\midrule",
        "Class & $n_c$ ($K_c$) Cal & Rank $k_c$ & Cutoff $1{-}q_c$ & Test $n$ ($K$) & Pooled Cov.\\ [Wilson CI] & Pooled Size & Mondrian Cov.\\ [Wilson CI] & Mondrian Size \\\\",
        "\\midrule",
    ]

    for cls in ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]:
        cd = cal_diag[cls]
        pr = pooled_pc.loc[cls]
        mr = mond_pc.loc[cls]
        tag = "$^\\dagger$" if pr["n_test"] <= 30 else ""
        btag = "$^*$" if cd["hits_exact_sample_max"] else ""
        lines.append(
            f"{cls}{tag} & {cd['n_conf_images']} ({cd['k_conf_clusters']}) & {cd['k_c_image']}{btag} & "
            f"{cd['prob_cutoff']:.4f} & {int(pr['n_test']):,} ({int(pr['n_test_clusters']):,}) & "
            f"{pr['coverage']*100:.2f}\\% [{pr['wilson_ci95_low']*100:.1f}, {pr['wilson_ci95_high']*100:.1f}] & {pr['mean_set_size']:.3f} & "
            f"{mr['coverage']*100:.2f}\\% [{mr['wilson_ci95_low']*100:.1f}, {mr['wilson_ci95_high']*100:.1f}] & {mr['mean_set_size']:.3f} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\multicolumn{9}{@{}l}{\\footnotesize $^\\dagger$Sparse test stratum ($n \\le 30$). $^*$Hits exact empirical maximum score $s_{(n_c)}$ ($n_c=9$ is the minimum feasible sample size for $\\alpha=0.10$); cluster-max quantiles are undefined ($K_c < 9$).}",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    (TAB_DIR / "tab10_mondrian_conformal.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_supplementary_tables() -> None:
    tab_s1 = """\\begin{table}[!htbp]
\\centering
\\caption{Evidence-classified training and temperature-scaling protocol for the frozen EfficientNet-B4 backbone. Where static YAML configuration defaults differ from executed Python training code and retained checkpoint metadata, only verified executed values are reported as active.}
\\label{tab:supp_classifier_protocol}
\\scriptsize
\\setlength{\\tabcolsep}{3.0pt}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{@{}p{2.7cm}p{7.1cm}p{3.6cm}@{}}
\\toprule
Component & Verified Executed Specification & Evidence Tag \\\\
\\midrule
Backbone \\& Head & \\texttt{timm} \\texttt{efficientnet\\_b4} ($17.9$M params); head: \\texttt{Flatten} $\\to$ \\texttt{BatchNorm1d(1792)} $\\to$ \\texttt{Dropout(0.3)} $\\to$ \\texttt{Linear(1792, 7)}. & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CHECKPOINT\\_METADATA} \\\\
Pretrained Provenance & ImageNet-1k weights (\\texttt{pretrained=True}, HuggingFace/\\texttt{timm} \\texttt{efficientnet\\_b4.ra2\\_in1k}). & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CONFIG}, \\texttt{VERIFIED\\_FROM\\_LOG} \\\\
Training Split \\& Sampling & ISIC~2019 lesion-group disjoint Train ($N=16{,}702$, $9{,}198$ groups); \\texttt{WeightedClassSampler}; batch size $32$, $8$ workers, \\texttt{pin\\_memory=True}. & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CHECKPOINT\\_METADATA} \\\\
Train Preprocessing & \\texttt{RandomResizedCrop(224, scale=(0.5,0.85), ratio=(0.9,1.1))}, H/V flips ($0.5$), ShiftScaleRotate ($0.1,0.15,90^\\circ,0.5$), Rotate90 ($0.3$), CoarseDropout, ColorJitter, CLAHE, Brightness/Contrast, HSV, GaussNoise, Blur, Compression, ImageNet norm. & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CHECKPOINT\\_METADATA} \\\\
Eval $A_0$ / $A_1$--$A_9$ Input & $A_0$: \\texttt{Resize(291)} $\\to$ \\texttt{CenterCrop(224)} $\\to$ ImageNet norm. $A_1$--$A_9$: deterministic operator $\\to$ \\texttt{Resize(224)} $\\to$ ImageNet norm (no second center crop). & \\texttt{VERIFIED\\_FROM\\_CODE} \\\\
Loss Function & \\texttt{CrossEntropyLoss(label\\_smoothing=0.1)} (YAML \\texttt{focal} loss option unimplemented in training loop). & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CHECKPOINT\\_METADATA} \\\\
Optimizer \\& LR & \\texttt{AdamW(lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight\\_decay=1e-4)}; YAML \\texttt{1e-3} unused. & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CHECKPOINT\\_METADATA} \\\\
LR Scheduler \\& EMA & \\texttt{CosineAnnealingLR(T\\_max=50, eta\\_min=0.0)}, step epoch-wise ($\\mathrm{lr}_{10}=9.045\\times 10^{-5}$); warmup ($5$ epochs) and EMA unimplemented/absent. & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CHECKPOINT\\_METADATA} \\\\
Precision \\& Determinism & AMP \\texttt{16-mixed}, gradient clip $1.0$, accumulate $1$, \\texttt{seed=42}, \\texttt{deterministic=True}. & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CHECKPOINT\\_METADATA} \\\\
Selection \\& Stopping & \\texttt{ModelCheckpoint} monitoring \\texttt{val/auroc} max (epoch $9$, $10$ epochs, $5{,}220$ steps; val AUROC $0.9331$, val Acc $71.34\\%$); \\texttt{EarlyStopping(patience=10)} had \\texttt{wait\\_count=0}; cloud halt trigger absent locally (limits reporting detail only; weights and eval splits preserved). & \\texttt{VERIFIED\\_FROM\\_CHECKPOINT\\_METADATA}, \\texttt{NOT\\_RECOVERABLE} (halt trigger) \\\\
Temperature Scaling & Post-hoc BFGS NLL minimization on frozen $A_0$ logits: $T_{\\mathrm{full}}=0.851796$ ($\\mathcal{D}_{\\mathrm{cal}}$) and $T_{\\mathrm{fit}}=0.855690$ ($\\mathcal{D}_{\\mathrm{fit}}$). & \\texttt{VERIFIED\\_FROM\\_CODE} \\\\
\\bottomrule
\\end{tabular}%
}
\\end{table}
"""
    (TAB_DIR / "tab_s1_classifier_protocol.tex").write_text(tab_s1, encoding="utf-8")

    tab_s2 = """\\begin{table}[!htbp]
\\centering
\\caption{Evidence-classified offline training and hyperparameter-selection protocol for the multi-head sequential action policy.}
\\label{tab:supp_policy_protocol}
\\scriptsize
\\setlength{\\tabcolsep}{3.0pt}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{@{}p{2.7cm}p{7.1cm}p{3.6cm}@{}}
\\toprule
Component & Verified Executed Specification & Evidence Tag \\\\
\\midrule
State \\& Action Input & $s_t \\in \\mathbb{R}^{28}$: [$p_t(7)$, $H(p_t)(1)$, $\\max p_t(1)$, $\\mathrm{margin}(1)$, $\\mathrm{history}(9)$, $\\mathrm{mask}(9)$] concatenated with 9-dim action one-hot ($37$ total inputs). & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CHECKPOINT\\_METADATA} \\\\
Trunk \\& Three Heads & Shared trunk \\texttt{Linear(37,64)} $\\to$ \\texttt{ReLU} $\\to$ \\texttt{Linear(64,64)} $\\to$ \\texttt{ReLU}; three linear scalar heads (\\texttt{64}$\\to$\\texttt{1}, squeezed): $\\Delta\\hat{Q}$ (quality), $\\Delta\\hat{C}$ (confidence), $\\mathrm{logit}_i$ (flip); $6{,}787$ params. & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CHECKPOINT\\_METADATA} \\\\
Supervision Targets & $\\Delta Q = (-\\log p_t(y)) - (-\\log p_{t+1}(y))$ (NLL reduction), $\\Delta C = \\max p_{t+1} - \\max p_t$, binary flip $I = \\mathbb{I}[\\hat{y}_{t+1} \\neq \\hat{y}_t]$ computed under recursive arithmetic averaging. & \\texttt{VERIFIED\\_FROM\\_CODE} \\\\
Trajectory Generation \\& Splits & Frozen per-action dev predictions ($N=21{,}645$: Train $16{,}702$, Val $3{,}542$, Cal $1{,}401$); NumPy RNG \\texttt{seed=42} samples $35$ subsets per image ($757{,}575$ states retained: $584{,}570$ Train, $123{,}970$ Val, $49{,}035$ Cal), expanding to $3{,}941{,}672$ Train and $835{,}912$ Val pairs (strictly disjoint from Test) with $\\mathrm{history}[k]+\\mathrm{mask}[k]=1$. & \\texttt{VERIFIED\\_FROM\\_CODE} \\\\
Objective \\& Optimizer & Unweighted ($1:1:1$) sum $\\mathrm{MSE}(\\Delta\\hat{Q},\\Delta Q)+\\mathrm{MSE}(\\Delta\\hat{C},\\Delta C)+\\mathrm{BCEWithLogits}(\\mathrm{logit}_i, I)$; \\texttt{AdamW(lr=1e-3, weight\\_decay=1e-4)}, Train BS $512$ (shuffled), Val BS $2{,}048$, $10$ epochs. & \\texttt{VERIFIED\\_FROM\\_CODE} \\\\
Checkpoint Selection & Best model saved whenever Validation total loss improved; no \\texttt{EarlyStopping} callback implemented in script. & \\texttt{VERIFIED\\_FROM\\_CODE} \\\\
Policy Seed \\& Loss Log & No explicit PyTorch weight-init / DataLoader seed set in policy training script; winning epoch index (out of $10$) and epoch loss curve not saved (prevents exact bit-for-bit retraining from scratch; frozen policy weights and dev states preserved). & \\texttt{NOT\\_RECOVERABLE} \\\\
Utility Weights ($\\lambda$) & $81$-point grid search on Val ($N=3{,}542$) minimizing summed Top-1 risk across $B=0..5$ $\\to (\\lambda_Q,\\lambda_C,\\lambda_I,\\lambda_K)=(5.0, 1.0, 0.5, 0.25)$. & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CONFIG} \\\\
Stopping Threshold ($\\tau$) & $50$-point grid $\\tau \\in [-2.0, 5.0]$ on Cal ($N=1{,}401$) minimizing summed Top-1 risk across $B=0..5$ $\\to \\tau = -0.4285714285714286$ ($-3/7$). & \\texttt{VERIFIED\\_FROM\\_CODE}, \\texttt{VERIFIED\\_FROM\\_CONFIG} \\\\
\\bottomrule
\\end{tabular}%
}
\\end{table}
"""
    (TAB_DIR / "tab_s2_policy_protocol.tex").write_text(tab_s2, encoding="utf-8")

    tab_s5 = """\\begin{table}[!htbp]
\\centering
\\caption{Canonical probability-source provenance and metric reconciliation across all manuscript tables and figures. Notice both $A_0$ Test arrays apply identical \\texttt{Resize(291)}$\\to$\\texttt{CenterCrop(224)} transforms and match across all $3{,}686/3{,}686$ discrete Top-1 predictions ($69.21\\%$ accuracy, $30.79\\%$ risk, $68.83\\%$ balanced accuracy); tiny continuous offsets ($\\max |\\Delta \\mathrm{logit}| = 0.006299$, $\\max |\\Delta p| = 0.001353$) reflect GPU cuDNN batch-size reduction order ($64$ vs.\\ $32$).}
\\label{tab:supp_prob_sources}
\\scriptsize
\\setlength{\\tabcolsep}{3.5pt}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{@{}p{3.6cm}p{4.3cm}p{5.5cm}@{}}
\\toprule
Analysis / Deliverable & Declared Canonical Source & Verified Key Metrics \\& Consistency Check \\\\
\\midrule
Global Baseline $A_0$ (Main Table~2, Raw \\& TS) & Single-view Test logits/probs ($\\mathrm{BS}=64$) + Calibration logits ($T_{\\mathrm{full}}=0.851796$) & Raw: Top-1 $69.21\\%$ ($2{,}551/3{,}686$), BAcc $68.83\\%$, AUROC $0.927355$, ECE $2.5203\\%$, NLL $0.8273$, Brier $0.4207$. Scaled ($T_{\\mathrm{full}}$): ECE $2.7088\\%$, NLL $0.8174$, Brier $0.4210$. \\\\
Split \\& Mondrian Conformal (Main Fig.~7, Tables~7--8) & Calibration logits split into $\\mathcal{D}_{\\mathrm{fit}}$ ($T_{\\mathrm{fit}}=0.855690$) \\& $\\mathcal{D}_{\\mathrm{conf}}$ ($N=703$); Test logits ($\\mathrm{BS}=64$) & Pooled $k^*=634, q^*=0.904267$, cov $90.67\\%$ ($3{,}342/3{,}686$), size $1.9249$; Mondrian cov $90.64\\%$, size $2.3896$ (unsplit interpolated quantiles excluded). \\\\
Sequential Multi-View \\& Spatial Ablation (Figs.~3--5, Tables~3--5) & Per-action Test tensor ($A_0$--$A_9$, $\\mathrm{BS}=32$) & Step $t=0$ matches single-view $A_0$ on $3{,}686/3{,}686$ Top-1 labels ($30.79\\%$ risk); continuous $A_0$ raw ECE $2.5383\\%$, AUROC $0.927356$ ($\\mathrm{mean}|\\Delta p|=5.78\\times 10^{-5}$). \\\\
Decision-Flip Instability (Fig.~6, Table~6) & Per-action Test rollouts ($M=14{,}815$ transitions, $B=5$) & AUROC $0.947302$, AP $0.517218$, Brier $0.041595$; $988$ consecutive Top-1 flips ($6.6689\\%$). \\\\
Perturbation Sensitivity (Fig.~8, Table~9) & Isolated Validation subset ($N=50$, $16$ transforms, frozen $A_0$) & Disjoint from Test ($0$ shared lesions/images); pairs unperturbed vs.\\ transformed $A_0$ predictions. \\\\
\\bottomrule
\\end{tabular}%
}
\\end{table}
"""
    (TAB_DIR / "tab_s5_probability_sources.tex").write_text(tab_s5, encoding="utf-8")


def main() -> None:
    sanitize_existing_tables()
    build_tab9_restricted_policy()
    build_tab10_mondrian_conformal()
    build_supplementary_tables()
    print("Clean-room aggregate tables generated in", TAB_DIR)


if __name__ == "__main__":
    main()
