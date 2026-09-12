"""
test_feature_extraction.py
==========================
Unit tests for the feature extraction functions in core_cv_logic.py:
- excess_green
- vein_density (Frangi)
- texture_feature (LBP)
- simulated_ndvi
"""

import numpy as np
import pytest
from core_cv_logic import (
    segment_leaf,
    excess_green,
    vein_density,
    texture_feature,
    simulated_ndvi,
)


class TestFeatureExtraction:
    """Test suite for color, vein, texture, and NDVI feature extractors."""

    def test_excess_green_healthy_vs_stressed(
        self, sample_healthy_leaf_rgb, sample_stressed_leaf_rgb
    ):
        """Healthy leaf with strong green pigments should have higher ExG than yellowish leaf."""
        mask_healthy = segment_leaf(sample_healthy_leaf_rgb)
        mask_stressed = segment_leaf(sample_stressed_leaf_rgb)

        exg_healthy = excess_green(sample_healthy_leaf_rgb, mask_healthy)
        exg_stressed = excess_green(sample_stressed_leaf_rgb, mask_stressed)

        assert isinstance(exg_healthy, float)
        assert isinstance(exg_stressed, float)
        assert exg_healthy > exg_stressed
        # Green-heavy leaf pixels typically have ExG > 100
        assert exg_healthy > 50.0

    def test_excess_green_mathematical_precision(self):
        """Verify exact mathematical calculation of ExG = 2G - R - B."""
        # 10x10 patch of known RGB: R=50, G=150, B=50 -> ExG = 2*150 - 50 - 50 = 200.0
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:] = [50, 150, 50]
        mask = np.ones((10, 10), dtype=np.uint8) * 255

        exg = excess_green(img, mask)
        assert np.isclose(exg, 200.0)

    def test_vein_density_output_structure_and_bounds(
        self, sample_healthy_leaf_rgb
    ):
        """Vein density must return a float score in [0, 1] and an RGB overlay matching input shape."""
        mask = segment_leaf(sample_healthy_leaf_rgb)
        score, overlay = vein_density(sample_healthy_leaf_rgb, mask)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == sample_healthy_leaf_rgb.shape
        assert overlay.dtype == np.uint8

    def test_vein_density_overlay_highlights_veins(
        self, sample_healthy_leaf_rgb
    ):
        """Vein overlay must highlight detected vein pixels with [0, 255, 128]."""
        mask = segment_leaf(sample_healthy_leaf_rgb)
        _, overlay = vein_density(sample_healthy_leaf_rgb, mask)

        # Check if overlay has green vein highlighted pixels [0, 255, 128]
        vein_pixels = np.all(overlay == [0, 255, 128], axis=-1)
        # Should have detected the drawn vein lines
        assert np.sum(vein_pixels) > 0

    def test_texture_feature_lbp(self, sample_healthy_leaf_rgb):
        """Texture feature must calculate the mean LBP score over the leaf mask."""
        mask = segment_leaf(sample_healthy_leaf_rgb)
        lbp_val = texture_feature(sample_healthy_leaf_rgb, mask)

        assert isinstance(lbp_val, float)
        assert lbp_val >= 0.0
        # 8-neighbor LBP values are bounded between 0 and 255
        assert lbp_val <= 255.0

    def test_simulated_ndvi_bounds_and_formula(
        self, sample_healthy_leaf_rgb
    ):
        """Simulated NDVI should be positive for green foliage and bounded in [-1.0, 1.0]."""
        mask = segment_leaf(sample_healthy_leaf_rgb)
        ndvi = simulated_ndvi(sample_healthy_leaf_rgb, mask)

        assert isinstance(ndvi, float)
        assert -1.0 <= ndvi <= 1.0
        assert ndvi > 0.0  # Healthy green leaf has positive NDVI

    def test_simulated_ndvi_deterministic_values(self):
        """Verify simulated NDVI formula: NIR = 0.5*R + 0.5*G; ndvi = (NIR - R)/(NIR + R)."""
        # Patch with R=40, G=160 -> NIR = 0.5*40 + 0.5*160 = 100
        # NDVI = (100 - 40) / (100 + 40 + 1e-6) = 60 / 140 = 0.42857...
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:] = [40, 160, 40]
        mask = np.ones((10, 10), dtype=np.uint8) * 255

        expected = (100.0 - 40.0) / (100.0 + 40.0)
        ndvi = simulated_ndvi(img, mask)
        assert np.isclose(ndvi, expected, atol=1e-3)
