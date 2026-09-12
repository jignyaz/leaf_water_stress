"""
test_segmentation.py
====================
Unit tests for the leaf segmentation stage in core_cv_logic.py.
"""

import numpy as np
import pytest
from core_cv_logic import segment_leaf


class TestLeafSegmentation:
    """Test suite for segment_leaf() HSV thresholding function."""

    def test_segment_leaf_output_properties(self, sample_healthy_leaf_rgb):
        """Segment leaf should return a single-channel uint8 binary mask of shape (H, W)."""
        mask = segment_leaf(sample_healthy_leaf_rgb)

        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        assert mask.shape == (512, 512)
        assert mask.dtype == np.uint8
        # Mask should only contain binary values: 0 (background) and 255 (leaf)
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255})

    def test_segment_leaf_identifies_green_leaf(self, sample_healthy_leaf_rgb):
        """Segment leaf should accurately identify the green leaf area with positive pixel count."""
        mask = segment_leaf(sample_healthy_leaf_rgb)
        leaf_pixels = np.sum(mask > 0)

        # Expected realistic leaf area (> 5000 pixels on 512x512)
        assert leaf_pixels > 5000
        # Should not cover the entire background
        assert leaf_pixels < (512 * 512)

    def test_segment_leaf_black_image_returns_zero_mask(self, sample_black_image):
        """Segment leaf should return an all-zero mask when given a completely black image."""
        mask = segment_leaf(sample_black_image)

        assert np.sum(mask) == 0
        assert np.all(mask == 0)

    def test_segment_leaf_non_green_image_returns_zero_mask(self, sample_non_leaf_image):
        """Segment leaf should reject non-green images (e.g. blue sky, red objects)."""
        mask = segment_leaf(sample_non_leaf_image)

        assert np.sum(mask) == 0

    @pytest.mark.parametrize(
        "rgb_color, expected_mask_val",
        [
            ([0, 255, 0], 255),      # Pure bright green -> inside HSV [25..100]
            ([45, 180, 50], 255),    # Natural cotton leaf green -> inside mask
            ([255, 0, 0], 0),        # Pure red -> outside (Hue ~0)
            ([0, 0, 255], 0),        # Pure blue -> outside (Hue ~120)
            ([255, 255, 255], 0),    # Pure white -> Saturation 0 < 40
            ([0, 0, 0], 0),          # Pure black -> Value 0 < 40
        ],
    )
    def test_segment_leaf_color_thresholds(self, rgb_color, expected_mask_val):
        """Tests individual HSV boundary conditions for specific RGB colors."""
        test_patch = np.zeros((50, 50, 3), dtype=np.uint8)
        test_patch[:] = rgb_color

        mask = segment_leaf(test_patch)
        assert np.all(mask == expected_mask_val)
