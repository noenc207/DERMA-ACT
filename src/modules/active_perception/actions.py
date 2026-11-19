"""
Canonical DERMA-ACT V1 Action Registry.

This is the SINGLE SOURCE OF TRUTH for all action-name mappings,
token conversions, and candidate-space semantics.

Global Action Space:  A0–A9  (10 actions)
Sequential Candidate Space:  A1–A9  (9 candidates; A0 is the initial
    observation and is never part of the candidate mask)

Candidate index 0  →  A1  →  center_zoom
Candidate index 8  →  A9  →  artifact_suppressed

Every module that needs action names MUST import from here.
Do NOT maintain independent action-name lists elsewhere.
"""

from __future__ import annotations
from .action_space import ObservationAction

# ── Canonical token-to-name mapping (A0–A9) ──────────────────────────────────
# This ordering is identical to the frozen scientific pipeline
# (scripts/run_derma_act_v0.py lines 104-120).

ACTION_NAMES: list[str] = [
    "global_view",            # A0
    "center_zoom",            # A1
    "border_zoom",            # A2
    "upper_region",           # A3
    "lower_region",           # A4
    "left_region",            # A5
    "right_region",           # A6
    "high_frequency_texture", # A7
    "color_suppressed",       # A8
    "artifact_suppressed",    # A9
]

# Canonical mapping from A-index to ObservationAction enum
# (same ordering as ACTIONS in run_derma_act_v0.py)
ACTION_ENUMS: list[ObservationAction] = [
    ObservationAction.KEEP_FULL,           # A0
    ObservationAction.ZOOM_CENTER,         # A1
    ObservationAction.ZOOM_BORDER,         # A2
    ObservationAction.TOP_REGION,          # A3
    ObservationAction.BOTTOM_REGION,       # A4
    ObservationAction.LEFT_REGION,         # A5
    ObservationAction.RIGHT_REGION,        # A6
    ObservationAction.TEXTURE_REGION,      # A7
    ObservationAction.COLOR_NORMALIZED,    # A8
    ObservationAction.ARTIFACT_SUPPRESSED, # A9
]

N_ACTIONS: int = len(ACTION_NAMES)           # 10
N_CANDIDATES: int = N_ACTIONS - 1            # 9  (A1–A9)


# ── Token conversion helpers ─────────────────────────────────────────────────

def action_token_to_index(token: str) -> int:
    """Convert a global action token string to its integer index.

    >>> action_token_to_index("A0")
    0
    >>> action_token_to_index("A9")
    9
    """
    if not token.startswith("A"):
        raise ValueError(f"Invalid action token: {token!r}")
    idx = int(token[1:])
    if idx < 0 or idx >= N_ACTIONS:
        raise ValueError(f"Action token {token!r} out of range [A0, A{N_ACTIONS-1}]")
    return idx


def action_index_to_token(index: int) -> str:
    """Convert a global action index (0–9) to its token string.

    >>> action_index_to_token(0)
    'A0'
    >>> action_index_to_token(9)
    'A9'
    """
    if index < 0 or index >= N_ACTIONS:
        raise ValueError(f"Action index {index} out of range [0, {N_ACTIONS-1}]")
    return f"A{index}"


def action_index_to_name(index: int) -> str:
    """Convert a global action index (0–9) to its human-readable name.

    >>> action_index_to_name(0)
    'global_view'
    >>> action_index_to_name(1)
    'center_zoom'
    """
    if index < 0 or index >= N_ACTIONS:
        raise ValueError(f"Action index {index} out of range [0, {N_ACTIONS-1}]")
    return ACTION_NAMES[index]


# ── Candidate-space conversion helpers ────────────────────────────────────────
# The candidate space contains A1–A9.  Candidate index 0 ↔ A1.

def candidate_index_to_token(cand_idx: int) -> str:
    """Convert a candidate-space index (0–8) to a global action token.

    >>> candidate_index_to_token(0)
    'A1'
    >>> candidate_index_to_token(8)
    'A9'
    """
    if cand_idx < 0 or cand_idx >= N_CANDIDATES:
        raise ValueError(
            f"Candidate index {cand_idx} out of range [0, {N_CANDIDATES-1}]"
        )
    return f"A{cand_idx + 1}"


def candidate_index_to_name(cand_idx: int) -> str:
    """Convert a candidate-space index (0–8) to a human-readable name.

    >>> candidate_index_to_name(0)
    'center_zoom'
    >>> candidate_index_to_name(8)
    'artifact_suppressed'
    """
    if cand_idx < 0 or cand_idx >= N_CANDIDATES:
        raise ValueError(
            f"Candidate index {cand_idx} out of range [0, {N_CANDIDATES-1}]"
        )
    return ACTION_NAMES[cand_idx + 1]


def candidate_index_to_global_index(cand_idx: int) -> int:
    """Convert a candidate-space index (0–8) to a global action index (1–9).

    >>> candidate_index_to_global_index(0)
    1
    >>> candidate_index_to_global_index(8)
    9
    """
    if cand_idx < 0 or cand_idx >= N_CANDIDATES:
        raise ValueError(
            f"Candidate index {cand_idx} out of range [0, {N_CANDIDATES-1}]"
        )
    return cand_idx + 1


def candidate_index_to_enum(cand_idx: int) -> ObservationAction:
    """Convert a candidate-space index (0–8) to its ObservationAction enum.

    >>> candidate_index_to_enum(0)
    <ObservationAction.ZOOM_CENTER: 1>
    """
    return ACTION_ENUMS[candidate_index_to_global_index(cand_idx)]
