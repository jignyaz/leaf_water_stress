"""
test_stress_scoring.py
======================
Unit tests for physiological stress score (PSS), canopy temperature estimation,
stress classification, and stress heatmap generation in core_cv_logic.py.
"""

import numpy as np
import pytest
from core_cv_logic import (
    segment_leaf,
    phys_stress,
    canopy_temp,
    classify_stress,
    stress_heatmap,
)


class TestStressScoring:
    """Test suite for stress scoring, classification, and visualization."""

    def test_phys_stress_exact_formula(self):
        """PSS must follow: 0.4*(1 - exg) + 0.35*vein + 0.25*texture."""
        exg, vein, tex = 0.5, 0.1, 4.0
        expected = 0.4 * (1 - 0.5) + 0.35 * 0.1 + 0.25 * 4.0
        # 0.4 * 0.5 = 0.20
        # 0.35 * 0.1 = 0.035
        # 0.25 * 4.0 = 1.00
        # expected = 1.235

        score = phys_stress(exg, vein, tex)
        assert np.isclose(score, expected, atol=1e-6)

    @pytest.mark.parametrize(
        "pss, expected_temp",
        [
            (-0.5, 31),
            (0.0, 31),
            (0.29, 31),
            (0.2999, 31),
            (0.3, 33),     # Boundary at 0.3
            (0.45, 33),
            (0.5999, 33),
            (0.6, 35),     # Boundary at 0.6
            (0.85, 35),
            (1.5, 35),
        ],
    )
    def test_canopy_temp_thresholds(self, pss, expected_temp):
        """Canopy temperature should map to 31, 33, or 35 depending on PSS level."""
        assert canopy_temp(pss) == expected_temp

    @pytest.mark.parametrize(
        "pss, expected_label",
        [
            (-0.1, "Healthy / No Stress"),
            (0.0, "Healthy / No Stress"),
            (0.299, "Healthy / No Stress"),
            (0.3, "Mild Stress"),
            (0.45, "Mild Stress"),
            (0.599, "Mild Stress"),
            (0.6, "Severe Stress"),
            (0.85, "Severe Stress"),
            (2.0, "Severe Stress"),
        ],
    )
    def test_classify_stress_labels(self, pss, expected_label):
        """classify_stress must return exact string labels according to PSS thresholds."""
        assert classify_stress(pss) == expected_label

    def test_stress_heatmap_shape_and_background(self, sample_healthy_leaf_rgb):
        """Heatmap must return an RGB uint8 image with dark background for unmasked pixels."""
        mask = segment_leaf(sample_healthy_leaf_rgb)
        heat = stress_heatmap(sample_healthy_leaf_rgb, mask)

        assert isinstance(heat, np.ndarray)
        assert heat.shape == sample_healthy_leaf_rgb.shape
        assert heat.dtype == np.uint8

        # Background pixels where mask == 0 must be set to [30, 30, 30]
        bg_pixels = heat[mask == 0]
        assert np.all(bg_pixels == [30, 30, 30])

        # Leaf pixels where mask > 0 should not be background
        leaf_pixels = heat[mask > 0]
        assert len(leaf_pixels) > 0
        assert not np.all(leaf_pixels == [30, 30, 30])
