"""
Canonical DERMA-ACT V1 Posterior Fusion.

Arithmetic mean fusion:
    p_new = (p_t * b + p_candidate) / (b + 1)

where b is the number of observations already fused (starts at 1 for A0).

This matches scripts/evaluate_v1_test.py line 246:
    p_t[c_idx] = (p_t[c_idx] * b + test_probs[c_idx, chosen + 1]) / (b + 1)
with b starting from 1 (since A0 is already incorporated).
"""
import numpy as np


def fuse_posterior(
    p_current: np.ndarray,
    p_candidate: np.ndarray,
    num_fused: int,
) -> np.ndarray:
    """Fuse a new candidate observation into the running posterior.

    Args:
        p_current: current fused posterior (7-class)
        p_candidate: new candidate's predicted probabilities (7-class)
        num_fused: number of observations already fused (>= 1, since A0 is always first)

    Returns:
        Updated posterior probabilities.

    Raises:
        ValueError: if num_fused < 1 (A0 must always be counted).
    """
    if num_fused < 1:
        raise ValueError(
            f"num_fused must be >= 1 (A0 is always the first observation). Got {num_fused}."
        )
    return (p_current * num_fused + p_candidate) / (num_fused + 1)
