"""
Canonical DERMA-ACT V1 State Construction.

28-dimensional state vector:
  [0:7]   - 7-class posterior probabilities
  [7]     - Shannon entropy (eps=1e-12)
  [8]     - confidence (max probability)
  [9]     - top-2 margin
  [10:19] - 9D history (binary: which A1-A9 have been acquired)
  [19:28] - 9D mask (binary: which A1-A9 are still available)

This matches scripts/build_v1_states.py lines 80-92
and scripts/evaluate_v1_test.py lines 185-189.
"""
import numpy as np
from research.derma_act.config import EPS


def entropy(p: np.ndarray) -> float:
    """Shannon entropy with canonical epsilon."""
    return float(-np.sum(p * np.log(p + EPS)))


def top2_margin(p: np.ndarray) -> float:
    """Difference between top-2 class probabilities."""
    s = np.sort(p)
    return float(s[-1] - s[-2])


def construct_state(
    p_t: np.ndarray,
    history: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Construct canonical 28-D state vector.

    Args:
        p_t: 7-class posterior probabilities
        history: 9D binary vector (acquired candidates)
        mask: 9D binary vector (available candidates)

    Returns:
        (state, entropy, confidence, margin)
    """
    ent = entropy(p_t)
    conf = float(np.max(p_t))
    margin = top2_margin(p_t)
    state = np.concatenate([p_t, [ent, conf, margin], history, mask]).astype(np.float32)
    return state, ent, conf, margin
