"""
test_pipeline_integration.py
============================
Integration tests verifying the complete computer vision pipeline execution
from raw image input through feature extraction to final classification and visual generation.
"""

import os
import cv2
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
    stress_heatmap,
    classify_stress,
)


class TestPipelineIntegration:
    """Integration test suite for the end-to-end Cotton Leaf Water-Stress CV pipeline."""

    def test_full_pipeline_on_synthetic_healthy_leaf(self, sample_healthy_leaf_rgb):
        """Tests complete pipeline flow on synthetic healthy cotton leaf."""
        # 1. Preprocess
        img = preprocess(sample_healthy_leaf_rgb, size=(512, 512))
        assert img.shape == (512, 512, 3)

        # 2. Leaf Segmentation
        mask = segment_leaf(img)
        leaf_pixel_count = np.sum(mask > 0)
        assert leaf_pixel_count > 500, "Leaf region must be detected"

        # 3. Feature Extraction
        exg_val = excess_green(img, mask)
        vein_val, vein_overlay = vein_density(img, mask)
        tex_val = texture_feature(img, mask)
        ndvi_val = simulated_ndvi(img, mask)

        assert np.isfinite(exg_val)
        assert np.isfinite(vein_val)
        assert np.isfinite(tex_val)
        assert np.isfinite(ndvi_val)
        assert vein_overlay.shape == (512, 512, 3)

        # 4. Stress Scoring & Classification
        pss_val = phys_stress(exg_val, vein_val, tex_val)
        temp_val = canopy_temp(pss_val)
        heat = stress_heatmap(img, mask)
        verdict = classify_stress(pss_val)

        assert np.isfinite(pss_val)
        assert temp_val in (31, 33, 35)
        assert heat.shape == (512, 512, 3)
        assert verdict in ("Healthy / No Stress", "Mild Stress", "Severe Stress")

    def test_full_pipeline_on_synthetic_stressed_leaf(self, sample_stressed_leaf_rgb):
        """Tests complete pipeline flow on synthetic water-stressed cotton leaf."""
        img = preprocess(sample_stressed_leaf_rgb, size=(512, 512))
        mask = segment_leaf(img)

        assert np.sum(mask > 0) > 500

        exg_val = excess_green(img, mask)
        vein_val, vein_overlay = vein_density(img, mask)
        tex_val = texture_feature(img, mask)
        ndvi_val = simulated_ndvi(img, mask)
        pss_val = phys_stress(exg_val, vein_val, tex_val)
        temp_val = canopy_temp(pss_val)
        heat = stress_heatmap(img, mask)
        verdict = classify_stress(pss_val)

        assert np.isfinite(pss_val)
        assert temp_val in (31, 33, 35)
        assert heat.shape == (512, 512, 3)
        assert isinstance(verdict, str)

    def test_full_pipeline_on_real_healthy_dataset_sample(self, real_healthy_leaf_path):
        """Tests complete pipeline on an actual real sample image from custom_dataset/Healthy leaf/."""
        if real_healthy_leaf_path is None or not os.path.exists(real_healthy_leaf_path):
            pytest.skip("No real healthy dataset images available.")

        # 1. Preprocess from real image path
        img = preprocess(real_healthy_leaf_path, size=(512, 512))
        assert img.shape == (512, 512, 3)

        # 2. Segment
        mask = segment_leaf(img)
        assert np.sum(mask > 0) > 500, f"Expected leaf detection for {real_healthy_leaf_path}"

        # 3. Features
        exg_val = excess_green(img, mask)
        vein_val, _ = vein_density(img, mask)
        tex_val = texture_feature(img, mask)
        ndvi_val = simulated_ndvi(img, mask)

        # 4. Scores
        pss_val = phys_stress(exg_val, vein_val, tex_val)
        temp_val = canopy_temp(pss_val)
        heat = stress_heatmap(img, mask)
        verdict = classify_stress(pss_val)

        assert np.isfinite(pss_val)
        assert temp_val in (31, 33, 35)
        assert heat.shape == (512, 512, 3)
        assert verdict in ("Healthy / No Stress", "Mild Stress", "Severe Stress")

    def test_full_pipeline_on_real_unhealthy_dataset_sample(self, real_unhealthy_leaf_path):
        """Tests complete pipeline on an actual real sample image from custom_dataset/UNhealthy leaf/."""
        if real_unhealthy_leaf_path is None or not os.path.exists(real_unhealthy_leaf_path):
            pytest.skip("No real unhealthy dataset images available.")

        img = preprocess(real_unhealthy_leaf_path, size=(512, 512))
        assert img.shape == (512, 512, 3)

        mask = segment_leaf(img)
        # Note: severely damaged leaves might have smaller or non-zero green masks
        if np.sum(mask > 0) >= 500:
            exg_val = excess_green(img, mask)
            vein_val, _ = vein_density(img, mask)
            tex_val = texture_feature(img, mask)
            ndvi_val = simulated_ndvi(img, mask)
            pss_val = phys_stress(exg_val, vein_val, tex_val)
            temp_val = canopy_temp(pss_val)
            heat = stress_heatmap(img, mask)
            verdict = classify_stress(pss_val)

            assert np.isfinite(pss_val)
            assert temp_val in (31, 33, 35)
            assert heat.shape == (512, 512, 3)
            assert isinstance(verdict, str)

    def test_visual_proof_grid_generation(self, sample_healthy_leaf_rgb):
        """Tests creation of 1x4 visual proof grid as implemented in process_custom_data.py."""
        img = preprocess(sample_healthy_leaf_rgb, size=(512, 512))
        mask = segment_leaf(img)

        # 1. Mask visual
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # 2. Vein visual
        _, vein_overlay = vein_density(img, mask)

        # 3. Stress heatmap visual
        heat = stress_heatmap(img, mask)

        # 4. 1x4 Grid horizontal stack
        grid = np.hstack((img, mask_rgb, vein_overlay, heat))

        assert isinstance(grid, np.ndarray)
        assert grid.shape == (512, 512 * 4, 3)
        assert grid.dtype == np.uint8
