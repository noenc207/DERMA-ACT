"""
Canonical DERMA-ACT V1 Preprocessing Transforms.

Two transform pipelines:
1. get_original_transform(): For raw images (A0 initial observation)
   - Resize(291) -> CenterCrop(224) -> Normalize -> ToTensorV2
   - Matches the canonical training/validation pipeline

2. get_view_transform(): For virtual observations A1-A9
   - Normalize -> ToTensorV2 ONLY
   - Views are ALREADY 224x224 from apply_action()
   - Do NOT re-apply Resize+CenterCrop
   - Matches scripts/run_derma_act_v0.py lines 91-100

This distinction is CRITICAL. Using get_val_transforms() on A1-A9 crops
applies Resize(291)->CenterCrop(224), which distorts the virtual views
by cropping 23% of the borders.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.core.constants import NORMALIZE_MEAN, NORMALIZE_STD

_IMAGE_SIZE = 224


def get_original_transform(image_size: int = _IMAGE_SIZE) -> A.Compose:
    """Transform for raw original images (before virtual observation generation).

    This is the standard val/inference pipeline:
        Resize(291) -> CenterCrop(224) -> Normalize -> ToTensorV2
    """
    return A.Compose([
        A.Resize(height=int(image_size * 1.3), width=int(image_size * 1.3)),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])


def get_view_transform() -> A.Compose:
    """Transform for virtual observation views (A1-A9) that are already 224x224.

    Only Normalize + ToTensorV2. No Resize, no CenterCrop.
    This matches the canonical scientific evaluator.
    """
    return A.Compose([
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])
