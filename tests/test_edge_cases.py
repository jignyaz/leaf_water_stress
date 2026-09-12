"""
test_edge_cases.py
==================
Unit tests for edge cases, numerical stability, boundary values,
and degenerate image inputs in the Cotton Leaf Water-Stress CV pipeline.
"""

import numpy as np
import pytest
from core_cv_logic import (
    preprocess,
    segment_leaf,
    excess_green,
    vein_density,
    texture_feature,
    simulated_ndvi,
    phys_stress,
    canopy_temp,
    classify_stress,
    stress_heatmap,
)


class TestEdgeCases:
    """Test suite for edge cases and numerical stability."""

    def test_single_pixel_leaf_mask(self):
        """Pipeline functions should handle tiny 1-pixel leaf masks gracefully without crashes."""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[256, 256] = [40, 180, 40]  # Single green pixel

        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[256, 256] = 255

        exg = excess_green(img, mask)
        assert np.isfinite(exg)

        score, overlay = vein_density(img, mask)
        assert np.isfinite(score)
        assert overlay.shape == (512, 512, 3)

        tex = texture_feature(img, mask)
        assert np.isfinite(tex)

        ndvi = simulated_ndvi(img, mask)
        assert np.isfinite(ndvi)

        heat = stress_heatmap(img, mask)
        assert heat.shape == (512, 512, 3)

    def test_uniform_color_leaf_heatmap_stability(self):
        """When leaf region has completely uniform ExG (vmin == vmax), heatmap normalization must not fail."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = [40, 160, 40]  # Uniform green across the whole patch
        mask = np.ones((100, 100), dtype=np.uint8) * 255

        heat = stress_heatmap(img, mask)
        assert isinstance(heat, np.ndarray)
        assert heat.shape == (100, 100, 3)
        assert not np.isnan(heat).any()

    def test_vein_density_empty_mask_zero_division_safety(self, sample_healthy_leaf_rgb):
        """Vein density calculation should safely return 0.0 with empty mask via epsilon guard."""
        empty_mask = np.zeros((512, 512), dtype=np.uint8)
        score, overlay = vein_density(sample_healthy_leaf_rgb, empty_mask)

        assert isinstance(score, float)
        assert score == 0.0
        assert overlay.shape == sample_healthy_leaf_rgb.shape

    def test_phys_stress_with_extreme_values(self):
        """PSS formula should handle zero and extreme values smoothly."""
        assert np.isclose(phys_stress(0.0, 0.0, 0.0), 0.4)
        assert np.isclose(phys_stress(1.0, 0.0, 0.0), 0.0)
        assert np.isclose(phys_stress(1.0, 1.0, 0.0), 0.35)

    def test_canopy_temp_extreme_bounds(self):
        """Canopy temp handles extreme low and high inputs without raising exceptions."""
        assert canopy_temp(-999.0) == 31
        assert canopy_temp(999.0) == 35

    def test_classify_stress_extreme_bounds(self):
        """classify_stress handles extreme negative or huge stress values safely."""
        assert classify_stress(-100.0) == "Healthy / No Stress"
        assert classify_stress(100.0) == "Severe Stress"
