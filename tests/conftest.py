"""
conftest.py
===========
Shared pytest fixtures and synthetic data generators for the Cotton Leaf Water-Stress tests.
"""

import os
import cv2
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def project_root() -> Path:
    """Returns the workspace root path."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def dataset_root(project_root) -> Path:
    """Returns the custom_dataset directory path."""
    return project_root / "CV Project" / "custom_dataset"


@pytest.fixture
def sample_healthy_leaf_rgb() -> np.ndarray:
    """
    Creates a synthetic 512x512 RGB image of a healthy green cotton leaf on a dark background.
    The green leaf pixels have high G value (e.g., [40, 160, 50]), falling within HSV green range.
    """
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Background: dark gray
    img[:] = [15, 15, 15]

    # Draw a central filled ellipse representing a leaf
    center = (256, 256)
    axes = (140, 190)
    angle = 15
    cv2.ellipse(img, center, axes, angle, 0, 360, (40, 165, 45), -1)

    # Add realistic vein lines inside the leaf
    cv2.line(img, (256, 90), (256, 420), (70, 210, 80), 2)
    cv2.line(img, (256, 180), (180, 240), (70, 210, 80), 2)
    cv2.line(img, (256, 180), (330, 230), (70, 210, 80), 2)
    cv2.line(img, (256, 280), (160, 340), (70, 210, 80), 2)
    cv2.line(img, (256, 280), (350, 330), (70, 210, 80), 2)

    return img


@pytest.fixture
def sample_stressed_leaf_rgb() -> np.ndarray:
    """
    Creates a synthetic 512x512 RGB image of a stressed, yellowish/wilted leaf.
    Lower excess green, higher red content (e.g., [140, 135, 30]).
    """
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [20, 20, 20]

    # Draw leaf ellipse
    center = (256, 256)
    axes = (130, 180)
    angle = -10
    cv2.ellipse(img, center, axes, angle, 0, 360, (140, 135, 30), -1)

    # Add prominent dried/stressed vein lines
    cv2.line(img, (256, 95), (256, 410), (180, 170, 50), 2)
    cv2.line(img, (256, 190), (170, 250), (180, 170, 50), 2)
    cv2.line(img, (256, 190), (340, 240), (180, 170, 50), 2)

    return img


@pytest.fixture
def sample_black_image() -> np.ndarray:
    """Creates a 512x512 pure black image."""
    return np.zeros((512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_non_leaf_image() -> np.ndarray:
    """Creates a 512x512 non-leaf blue/sky-colored image."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [30, 80, 220]  # Blue RGB
    return img


@pytest.fixture
def real_healthy_leaf_path(dataset_root) -> str | None:
    """Returns the absolute path to a real healthy leaf image from custom_dataset, if exists."""
    folder = dataset_root / "Healthy leaf"
    if folder.is_dir():
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                return str(folder / f)
    return None


@pytest.fixture
def real_unhealthy_leaf_path(dataset_root) -> str | None:
    """Returns the absolute path to a real unhealthy leaf image from custom_dataset, if exists."""
    folder = dataset_root / "UNhealthy leaf"
    if folder.is_dir():
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                return str(folder / f)
    return None


@pytest.fixture
def temp_image_path(tmp_path, sample_healthy_leaf_rgb) -> str:
    """Saves the synthetic healthy leaf to a temporary JPEG file and returns its path."""
    file_path = tmp_path / "temp_healthy_leaf.jpg"
    # OpenCV writes in BGR
    bgr = cv2.cvtColor(sample_healthy_leaf_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(file_path), bgr)
    return str(file_path)
