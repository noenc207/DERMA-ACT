# Experiment Provenance (`DERMA-ACT V2`)

- **Frozen Scientific Record:** `PAPER_DRAFT_Q2_V3_2` (preserved byte-for-byte in workspace).
- **Current Pre-Submission Candidate:** `PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION` (*Biomedical Signal Processing and Control* candidate revision with resolved author metadata and Declarations).
- **Version-Controlled Pure Aggregate Result Ledgers (under `research/derma_act/`):**
  - `evidence/v1/FINAL_RISK_COST_TABLE.csv`: V1 vs fixed policy trajectories (`B = 1..5`).
  - `p5_4_independent_policy_validation/P5_4_BENCHMARK_ANALYSIS.json`: Exact McNemar & Holm-Bonferroni statistics (`p_adj = 0.3185`, reported `0.319` at `B = 5` vs `CAL-Stop`).
  - `p5_5_final_claim_closure/decision_flip_cluster_results.json` & `minority_class_audit.json`: Decision-flip instability head (`M = 14,815`, `988` flips, `AUROC = 0.9473`, `AP = 0.5172`, `Brier = 0.0416`) & per-class sensitivity trade-offs.
  - `p5_6_action_quality_audit/`: `A7` (`99.97%` saturation, `25.69%` zero-collapse) and `A9` (`11.55%` masked area) operator forensics.
  - `p7_1_conformal_quantile_audit/P7_1_GROUP_COVERAGE_RECONCILIATION.csv`: LAC split-conformal quantile reconciliation (`q* = 0.904267`, `k* = 634`, `90.67%` marginal coverage, `1.92` mean set size, `AKIEC` `74.21%`).
  - `p8_1_final_submission_audit/phase_f_restricted_policy/` & `phase_g_mondrian_conformal/`: Post-hoc 5-spatial-action masking (`26.21%` error, `3.44` mean views, `p_adj = 1.000` vs `CAL-Stop`) & image-level Mondrian calibration (`AKIEC` `70.59%`, `156/221` test images from `n_c = 30`, `K_c = 17` in `D_conf`).
- **MLflow Tracking DB Disposition:** `_quarantined/old_data/mlflow.db` (7 early 1-epoch test runs, `0` logged metrics) is unrelated to frozen evaluation results, contains local workstation URIs, and is excluded from new commits (excluded per `LICENSE_STATUS.md` and `.gitignore`).
- **Superseded / Quarantined / Invalid Work Excluded:** Buggy `B = 0` HIBA logs, `invalid_cohort_picsum/`, and historical PAD-UFES-20 external collapse logs are omitted from repository claims.
