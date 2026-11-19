"""
DERMA-ACT V1 Canonical Configuration.

Single source of truth for all DERMA-ACT parameters.
All live/evaluation code should derive values from this module.
"""
from pathlib import Path
import json

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Frozen artifact paths
CLASSIFIER_PATH = PROJECT_ROOT / 'checkpoints' / 'baseline_v2' / 'clean_run_001' / 'epoch=09-val_auroc=0.0000.ckpt'
POLICY_PATH = PROJECT_ROOT / 'research' / 'derma_act' / 'runs' / 'derma_act_v1' / 'policy' / 'policy.pt'
LAMBDA_PATH = PROJECT_ROOT / 'research' / 'derma_act' / 'runs' / 'derma_act_v1' / 'policy' / 'lambda_selection.json'
TAU_PATH = PROJECT_ROOT / 'research' / 'derma_act' / 'runs' / 'derma_act_v1' / 'policy' / 'stopping_threshold.json'

# Expected SHA256 hashes
EXPECTED_HASHES = {
    'classifier': '0d14579d668d02781874cf91ad493fac4bf33ed53e3be928e3a7f9bb856ebbbb',
    'policy': '32a098c8e2aa958959122cb0292eff0cc8932ea4dac4c2599f3dd37ed2587c95',
    'lambda': '3662d80eacb9ca39276697ff5f9d043d3774794a9589f316b81b2c0fdf0dea05',
    'tau': 'dc13a238e16a678bf153beb8b2cbbb7341f8360f8e8d0f8172ee68bf67eaf8ef',
}

# Dimensions
STATE_DIM = 28      # 7 probs + 3 uncertainty + 9 history + 9 mask
CANDIDATE_DIM = 9   # A1-A9
N_CLASSES = 7
N_ACTIONS = 10      # A0-A9
BUDGET = 5          # max non-A0 observations
HIDDEN_DIM = 64     # policy hidden dimension
IMAGE_SIZE = 224

# Lambda coefficients (frozen)
LAMBDA_Q = 5.0
LAMBDA_C = 1.0
LAMBDA_I = 0.5
LAMBDA_K = 0.25

# Stopping threshold (frozen)
TAU = -0.4286

# Cost convention
ACTION_COST = 1.0

# Canonical epsilon for entropy
EPS = 1e-12

# Canonical class names (order matches classifier output)
CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']

# Display names
DISPLAY_CLASSES = {
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'bcc': 'BCC',
    'akiec': 'AKIEC',
    'bkl': 'BKL',
    'df': 'DF',
    'vasc': 'VASC',
}

# Action display mapping (derived from canonical registry)
ACTION_DISPLAY = {
    'A0': 'Global View',
    'A1': 'Center Zoom',
    'A2': 'Border Zoom',
    'A3': 'Upper Region',
    'A4': 'Lower Region',
    'A5': 'Left Region',
    'A6': 'Right Region',
    'A7': 'High Frequency Texture',
    'A8': 'Color Suppressed',
    'A9': 'Artifact Suppressed',
}


def load_lambda():
    """Load frozen lambda coefficients from lambda_selection.json."""
    with open(LAMBDA_PATH) as f:
        cfg = json.load(f)
    best = cfg['best_lambda']
    return best['lambda_Q'], best['lambda_C'], best['lambda_I'], best['lambda_K']


def load_tau():
    """Load frozen stopping threshold from stopping_threshold.json."""
    with open(TAU_PATH) as f:
        return json.load(f)['best_tau']
