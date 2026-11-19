import torch
import math

def transform_instability(logit):
    """
    Transforms the raw instability logit output from the policy's third head
    into an instability probability using the sigmoid function.
    """
    if isinstance(logit, torch.Tensor):
        return torch.sigmoid(logit)
    else:
        # Handle scalar float natively
        if logit < -100:
            return 0.0
        return 1.0 / (1.0 + math.exp(-logit))

def compute_utility(dq, dc, instability_logit, l_q, l_c, l_i, l_k):
    """
    Computes the canonical DERMA-ACT V1 utility value for a candidate action.
    U = l_q * dq + l_c * dc - l_i * sigmoid(instability_logit) - l_k * 1.0
    """
    instability_prob = transform_instability(instability_logit)
    return l_q * dq + l_c * dc - l_i * instability_prob - l_k * 1.0
