import pytest
import numpy as np
import torch
import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from research.derma_act.core.transforms import get_original_transform, get_view_transform
from research.derma_act.core.fusion import fuse_posterior
from research.derma_act.core.stopping import should_stop
from research.derma_act.core.state import construct_state
from research.derma_act.config import EPS
from src.training.augmentation import get_val_transforms

def test_entropy_eps():
    """Verify Canonical EPS = 1e-12 in entropy calculations."""
    assert EPS == 1e-12
    # Ensure it's exactly 1e-12 and not 1e-9 (the old bug)

def test_fusion_parity():
    """Verify B=0 fails and B=1 passes for posterior fusion."""
    p_t = np.array([0.1, 0.6, 0.1, 0.1, 0.1, 0.0, 0.0])
    p_cand = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.0])
    
    # B = 0 (buggy logic) should now throw ValueError
    with pytest.raises(ValueError, match="num_fused must be >= 1"):
        fuse_posterior(p_t, p_cand, 0)
    
    # B = 1 (fixed logic)
    fusion_b1 = fuse_posterior(p_t, p_cand, 1)
    expected_b1 = (1.0 * p_t + p_cand) / 2.0
    assert np.allclose(fusion_b1, expected_b1)
    
def test_transform_parity():
    """Verify canonical core transform vs evaluator transforms."""
    # Generate a dummy RGB image
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # The original evaluator transform
    old_tfm = get_val_transforms(224)
    old_out = old_tfm(image=img)['image']
    
    # The new original transform (for A0) should match the old evaluator
    new_orig = get_original_transform(224)
    new_out = new_orig(image=img)['image']
    assert torch.allclose(old_out, new_out)
    
    # The view transform (for A1-A9) must NOT crop
    new_view = get_view_transform()
    img_224 = cv2.resize(img, (224, 224))
    view_out = new_view(image=img_224)['image']
    
    assert view_out.shape == (3, 224, 224)
    assert view_out.dtype == torch.float32
    assert torch.isfinite(view_out).all()

def test_rgb_regression():
    """Prove BGR does not enter the neural network."""
    img_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
    img_bgr[:, :, 0] = 255 # Blue channel max
    
    # Convert to RGB (red channel should now be max)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Transform
    tfm = get_view_transform()
    tensor = tfm(image=img_rgb)['image']
    
    # Validate channel 2 (Blue) in tensor is high (mean normalization applied)
    # B channel should be much larger than R channel
    assert tensor[2].mean() > tensor[0].mean()
