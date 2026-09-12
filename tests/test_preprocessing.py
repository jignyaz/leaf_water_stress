"""
test_preprocessing.py
=====================
Unit tests for the image preprocessing stage in core_cv_logic.py.
"""

import os
import cv2
import numpy as np
import pytest
from core_cv_logic import preprocess


class TestPreprocessing:
    """Test suite for preprocess() function."""

    def test_preprocess_with_rgb_array(self, sample_healthy_leaf_rgb):
        """Preprocess should accept an RGB array and return a uint8 array of target size (512, 512, 3)."""
        input_img = cv2.resize(sample_healthy_leaf_rgb, (300, 400))
        result = preprocess(input_img, size=(512, 512))

        assert isinstance(result, np.ndarray)
        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8

    def test_preprocess_custom_size(self, sample_healthy_leaf_rgb):
        """Preprocess should correctly resize to custom target dimensions."""
        custom_size = (256, 128)  # width=256, height=128
        result = preprocess(sample_healthy_leaf_rgb, size=custom_size)

        assert result.shape == (128, 256, 3)

    def test_preprocess_does_not_mutate_input(self, sample_healthy_leaf_rgb):
        """Preprocess should not mutate the original numpy array."""
        original_copy = sample_healthy_leaf_rgb.copy()
        result = preprocess(sample_healthy_leaf_rgb, size=(256, 256))

        # Original array should remain unchanged
        np.testing.assert_array_equal(sample_healthy_leaf_rgb, original_copy)
        assert result.shape != sample_healthy_leaf_rgb.shape

    def test_preprocess_from_valid_file_path(self, temp_image_path):
        """Preprocess should load from a valid image path, convert BGR to RGB, and resize."""
        result = preprocess(temp_image_path, size=(512, 512))

        assert isinstance(result, np.ndarray)
        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8

    def test_preprocess_nonexistent_file_raises_value_error(self):
        """Preprocess should raise ValueError when given a non-existent file path."""
        bad_path = "non_existent_leaf_img_xyz_9999.jpg"
        with pytest.raises(ValueError, match="Cannot read image"):
            preprocess(bad_path)

    def test_preprocess_corrupted_file_raises_value_error(self, tmp_path):
        """Preprocess should raise ValueError when given a corrupt/empty file."""
        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_text("This is not a valid image file")

        with pytest.raises(ValueError, match="Cannot read image"):
            preprocess(str(corrupt_file))
