"""Standalone DERMA-ACT V2 repository verification and hygiene smoke tests."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# Explicitly enumerate version-controlled DERMA-ACT V2 release targets so the smoke
# check runs cleanly both inside a multi-artifact workspace and in a clean Git clone.
RELEASE_PREFIXES = (
    "docs/",
    "provenance/",
    "configs/",
    "research/derma_act/core/",
    "scripts/figure_sources/",
    "supplementary/tables/",
    "manuscripts/current/PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/tables/",
    "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/tables/",
)
RELEASE_FILES = (
    "README.md",
    "CITATION.cff",
    "LICENSE_STATUS.md",
    "pyproject.toml",
    ".env.example",
    "data/aggregate/FIGURE_4_V1_1_DATA.csv",
    "src/modules/active_perception/actions.py",
    "src/modules/active_perception/policy.py",
    "src/modules/active_perception/utility.py",
    "research/derma_act/__init__.py",
    "research/derma_act/config.py",
    "research/derma_act/evidence/v1/FINAL_RISK_COST_TABLE.csv",
    "research/derma_act/p5_4_independent_policy_validation/P5_4_BENCHMARK_ANALYSIS.json",
    "research/derma_act/p5_5_final_claim_closure/decision_flip_cluster_results.json",
    "research/derma_act/p5_5_final_claim_closure/minority_class_audit.json",
    "research/derma_act/p5_6_action_quality_audit/P5_6_A7_COHORT_SUMMARY.json",
    "research/derma_act/p5_6_action_quality_audit/P5_6_A9_COHORT_SUMMARY.json",
    "research/derma_act/p5_6_action_quality_audit/P5_6_ACTION_PREDICTION_METRICS.csv",
    "research/derma_act/p5_6_action_quality_audit/P5_6_ACTION_QUALITY_REGISTRY.csv",
    "research/derma_act/p5_6_action_quality_audit/P5_6_POLICY_ACTION_FREQUENCIES.csv",
    "research/derma_act/p5_6_action_quality_audit/P5_6_TRANSITION_MATRICES.json",
    "research/derma_act/p7_1_conformal_quantile_audit/P7_1_GROUP_COVERAGE_RECONCILIATION.csv",
    "research/derma_act/p8_1_final_submission_audit/phase_f_restricted_policy/phase_f_comparison_table.csv",
    "research/derma_act/p8_1_final_submission_audit/phase_f_restricted_policy/phase_f_config.json",
    "research/derma_act/p8_1_final_submission_audit/phase_f_restricted_policy/phase_f_feasibility_and_results.json",
    "research/derma_act/p8_1_final_submission_audit/phase_f_restricted_policy/phase_f_paired_tests.csv",
    "research/derma_act/p8_1_final_submission_audit/phase_g_mondrian_conformal/phase_g_config.json",
    "research/derma_act/p8_1_final_submission_audit/phase_g_mondrian_conformal/phase_g_feasibility_and_results.json",
    "research/derma_act/p8_1_final_submission_audit/phase_g_mondrian_conformal/phase_g_mondrian_per_class.csv",
    "research/derma_act/p8_1_final_submission_audit/phase_g_mondrian_conformal/phase_g_mondrian_summary.csv",
    "research/derma_act/runs/derma_act_v1/policy/lambda_selection.json",
    "research/derma_act/runs/derma_act_v1/policy/stopping_threshold.json",
    "scripts/build_figures.py",
    "scripts/evaluate_v1_test.py",
    "scripts/generate_graphical_abstract.py",
    "scripts/generate_tables.py",
    "scripts/run_p7_1_conformal_audit.py",
    "scripts/run_phase_f_restricted_policy.py",
    "scripts/run_phase_g_mondrian_conformal.py",
    "tests/test_canonical_math.py",
    "tests/test_release_smoke.py",
    "supplementary/HIGHLIGHTS.txt",
    "supplementary/README.md",
    "manuscripts/current/PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/main.tex",
    "manuscripts/current/PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/references.bib",
    "manuscripts/current/PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/supplementary/supplementary_materials.tex",
    "manuscripts/current/PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/HIGHLIGHTS_V3_2_REF.txt",
    "manuscripts/current/PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/V3_2_1_PLACEHOLDER_AUDIT_UPDATED.json",
    "manuscripts/current/PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/V3_2_1_STATISTICAL_VERIFICATION.json",
    "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/main.tex",
    "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/references.bib",
    "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/supplementary/supplementary_materials.tex",
    "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/HIGHLIGHTS_V3_2_REF.txt",
    "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/V3_2_1_PLACEHOLDER_AUDIT_UPDATED.json",
    "PAPER_DRAFT_Q2_V3_2_1_PRE_SUBMISSION/V3_2_1_STATISTICAL_VERIFICATION.json",
)


def _iter_release_files():
    seen = set()
    for rel in RELEASE_FILES:
        p = REPO_ROOT / rel
        if p.is_file() and rel not in seen:
            seen.add(rel)
            yield rel, p
    for pref in RELEASE_PREFIXES:
        base = REPO_ROOT / pref
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    rel = p.relative_to(REPO_ROOT).as_posix()
                    if rel not in seen:
                        seen.add(rel)
                        yield rel, p


def test_frozen_policy_configs_and_architecture():
    from research.derma_act.config import BUDGET, CLASS_NAMES, LAMBDA_C, LAMBDA_I, LAMBDA_K, LAMBDA_Q, STATE_DIM, TAU
    from src.modules.active_perception.policy import MultiHeadActionPolicy

    assert len(CLASS_NAMES) == 7
    assert STATE_DIM == 28
    assert BUDGET == 5
    assert (LAMBDA_Q, LAMBDA_C, LAMBDA_I, LAMBDA_K) == (5.0, 1.0, 0.5, 0.25)
    assert TAU == -0.4286

    lam_cfg = json.loads((REPO_ROOT / "research/derma_act/runs/derma_act_v1/policy/lambda_selection.json").read_text(encoding="utf-8"))
    tau_cfg = json.loads((REPO_ROOT / "research/derma_act/runs/derma_act_v1/policy/stopping_threshold.json").read_text(encoding="utf-8"))
    assert lam_cfg["best_lambda"]["lambda_Q"] == 5.0
    assert abs(tau_cfg["best_tau"] - (-3.0 / 7.0)) < 1e-12

    model = MultiHeadActionPolicy(state_dim=28, action_dim=9, hidden_dim=64)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 6787, f"Expected 6787 policy params, got {n_params}"

    x = torch.zeros((2, 28), dtype=torch.float32)
    action_oh = torch.eye(9, dtype=torch.float32)[:2]
    dq_hat, dc_hat, i_logit = model(x, action_oh)
    assert dq_hat.shape == (2,)
    assert dc_hat.shape == (2,)
    assert i_logit.shape == (2,)


def test_no_credentials_machine_paths_or_forbidden_artifacts():
    win_drive_pat = re.compile(r"(?i)[a-z]:[\\/]" + "trustworthy")
    user_home_pat = re.compile(r"(?i)(c:[\\/]" + "users[\\/]|/home/|/users/)[a-z0-9_.-]+")
    tok_pat = re.compile(r"(gh[pousr]_[A-Za-z0-9_]{10,}|github_" + r"pat_[A-Za-z0-9_]{10,}|AI" + r"za[0-9A-Za-z_-]{20,}|BEGIN (RSA |EC |)PRIVATE KEY)")

    checked = 0
    for rel, p in _iter_release_files():
        checked += 1
        assert not rel.endswith((".db", ".ckpt", ".pt", ".npz", ".key", ".pem", ".jpg", ".jpeg")), f"Forbidden artifact type: {rel}"
        if p.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".toml", ".cff", ".txt", ".tex"}:
            txt = p.read_text(encoding="utf-8", errors="replace")
            assert not tok_pat.search(txt), f"Credential pattern in {rel}"
            assert not win_drive_pat.search(txt), f"Hardcoded workstation path in {rel}"
            assert not user_home_pat.search(txt), f"Local username path in {rel}"
    assert checked >= 50
