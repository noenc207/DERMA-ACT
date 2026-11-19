"""
Canonical DERMA-ACT V1 Stopping Rule.

Risk-aware utility-threshold stopping:
    STOP if max_a U(s_t, a) < tau

This matches scripts/evaluate_v1_test.py lines 212-214:
    best_u = U_hat_all.max(axis=1)
    should_stop = best_u < tau
"""


def should_stop(max_utility: float, tau: float) -> bool:
    """Determine whether to stop acquiring virtual observations.

    Args:
        max_utility: maximum utility across all available candidate actions
        tau: frozen stopping threshold

    Returns:
        True if the agent should stop (utility below threshold).
    """
    return max_utility < tau
