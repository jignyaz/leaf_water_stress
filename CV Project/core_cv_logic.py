"""
core_cv_logic.py
================
Shared traditional-CV functions for the Cotton Leaf Water-Stress project.
Imported by both process_custom_data.py and app.py.

Libraries: cv2, numpy, skimage  —  NO deep learning.
"""

import cv2
import numpy as np
import matplotlib.cm as cm
from skimage.filters import frangi
from skimage.feature import local_binary_pattern


# ──────────────────────────────────────────────────────────────
# 1. PREPROCESSING
# ──────────────────────────────────────────────────────────────
def preprocess(path_or_array, size=(512, 512)):
    """
    Accept either a file path (str) or a numpy array already in RGB.
    Returns a uint8 RGB image resized to `size`.
    """
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array)
        if img is None:
            raise ValueError(f"Cannot read image: {path_or_array}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = path_or_array.copy()

    img = cv2.resize(img, size)
    return img


# ──────────────────────────────────────────────────────────────
# 2. LEAF SEGMENTATION
# ──────────────────────────────────────────────────────────────
def segment_leaf(img):
    """HSV green-range binary mask."""
    hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (25, 40, 40), (100, 255, 255))
    return mask


# ──────────────────────────────────────────────────────────────
# 3. EXCESS GREEN INDEX
# ──────────────────────────────────────────────────────────────
def excess_green(img, mask):
    """ExG = 2G − R − B averaged over leaf pixels."""
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exg = 2 * G.astype(float) - R.astype(float) - B.astype(float)
    return float(np.mean(exg[mask > 0]))


# ──────────────────────────────────────────────────────────────
# 4. VEIN DENSITY  (Frangi filter)
# ──────────────────────────────────────────────────────────────
def vein_density(img, mask):
    """Frangi vein density within the leaf mask.  Returns (score, overlay_BGR)."""
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v     = frangi(gray / 255.0)
    veins = (v > 0.05).astype(np.uint8)
    score = float(np.sum(veins & (mask > 0)) / (np.sum(mask > 0) + 1e-6))

    # Build a colour overlay for the app (RGB)
    overlay = img.copy()
    overlay[veins == 1] = [0, 255, 128]   # bright green veins
    return score, overlay


# ──────────────────────────────────────────────────────────────
# 5. TEXTURE  (LBP)
# ──────────────────────────────────────────────────────────────
def texture_feature(img, mask):
    """Mean LBP value inside the leaf mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, 8, 1)
    return float(np.mean(lbp[mask > 0]))


# ──────────────────────────────────────────────────────────────
# 6. SIMULATED NDVI
# ──────────────────────────────────────────────────────────────
def simulated_ndvi(img, mask):
    """Simulated NDVI (no actual NIR sensor available)."""
    R   = img[:, :, 0].astype(float)
    G   = img[:, :, 1].astype(float)
    NIR = 0.5 * R + 0.5 * G
    ndvi = (NIR - R) / (NIR + R + 1e-6)
    return float(np.mean(ndvi[mask > 0]))


# ──────────────────────────────────────────────────────────────
# 7. PHYSIOLOGICAL STRESS SCORE
# ──────────────────────────────────────────────────────────────
def phys_stress(exg, vein, texture):
    """PSS weighted formula (preserved from mid-term)."""
    return 0.4 * (1 - exg) + 0.35 * vein + 0.25 * texture


# ──────────────────────────────────────────────────────────────
# 8. CANOPY TEMPERATURE ESTIMATE
# ──────────────────────────────────────────────────────────────
def canopy_temp(pss):
    """Heuristic canopy temperature from PSS level."""
    if pss < 0.3:
        return 31
    elif pss < 0.6:
        return 33
    else:
        return 35


# ──────────────────────────────────────────────────────────────
# 9. STRESS HEATMAP
# ──────────────────────────────────────────────────────────────
def stress_heatmap(img, mask):
    """
    Generate a colour heatmap of per-pixel ExG stress intensity.
    Red = low ExG (stressed), Green = high ExG (healthy).
    Returns an RGB image.
    """
    R = img[:, :, 0].astype(float)
    G = img[:, :, 1].astype(float)
    B = img[:, :, 2].astype(float)
    exg = 2 * G - R - B

    # Normalise to [0, 1] within the leaf region
    exg_masked = np.where(mask > 0, exg, np.nan)
    vmin, vmax = np.nanmin(exg_masked), np.nanmax(exg_masked)
    norm = np.clip((exg - vmin) / (vmax - vmin + 1e-6), 0, 1)

    # Use matplotlib's RdYlGn colormap (red=stressed, green=healthy)
    # cm(norm) returns an RGBA float array in [0,1]; convert to uint8 RGB
    rdylgn = cm.get_cmap('RdYlGn')
    heat = (rdylgn(norm)[:, :, :3] * 255).astype(np.uint8)  # drop alpha channel
    heat[mask == 0] = [30, 30, 30]         # dark background for non-leaf pixels
    return heat


# ──────────────────────────────────────────────────────────────
# 10. CLASSIFY FINAL SCORE
# ──────────────────────────────────────────────────────────────
def classify_stress(pss):
    """
    Map the raw PSS to a human-readable stress level.
    Thresholds calibrated for typical PSS range.
    """
    if pss < 0.3:
        return "Healthy / No Stress"
    elif pss < 0.6:
        return "Mild Stress"
    else:
        return "Severe Stress"
