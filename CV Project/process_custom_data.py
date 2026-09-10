"""
process_custom_data.py
======================
Processes the custom cotton-leaf dataset for final evaluation AND 
generates 1x4 visual proof grids for every processed leaf.
"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi
from skimage.feature import local_binary_pattern

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_ROOT  = "custom_dataset"
VISUALS_ROOT  = "output_visuals"  # New directory for saved images
FOLDERS       = {
    "Healthy leaf":   "healthy",
    "UNhealthy leaf": "unhealthy",
}
OUTPUT_CSV    = "custom_dataset_results.csv"

# ─────────────────────────────────────────────
# SETUP DIRECTORIES
# ─────────────────────────────────────────────
if not os.path.exists(VISUALS_ROOT):
    os.makedirs(VISUALS_ROOT)
for folder_name in FOLDERS.keys():
    os.makedirs(os.path.join(VISUALS_ROOT, folder_name), exist_ok=True)

# ─────────────────────────────────────────────
# CORE CV FUNCTIONS  (preserved exactly)
# ─────────────────────────────────────────────

def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"cv2.imread returned None — file unreadable: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    return img

def segment_leaf(img):
    hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (25, 40, 40), (100, 255, 255))
    return mask

def excess_green(img, mask):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    exg = 2 * G - R - B
    return np.mean(exg[mask > 0])

def vein_density(img, mask):
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v     = frangi(gray / 255.0)
    veins = v > 0.05
    return np.sum(veins & (mask > 0)) / (np.sum(mask > 0) + 1e-6)

def texture_feature(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, 8, 1)
    return np.mean(lbp[mask > 0])

def simulated_ndvi(img, mask):
    R   = img[:, :, 0].astype(float)
    G   = img[:, :, 1].astype(float)
    NIR = 0.5 * R + 0.5 * G
    ndvi = (NIR - R) / (NIR + R + 1e-6)
    return np.mean(ndvi[mask > 0])

def phys_stress(exg, vein, texture):
    return 0.4 * (1 - exg) + 0.35 * vein + 0.25 * texture

def canopy_temp(pss):
    if pss < 0.3: return 31
    elif pss < 0.6: return 33
    else: return 35

# ─────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────
raw           = []
processed     = 0
skipped       = 0

print("=" * 55)
print("  Cotton Leaf Water-Stress — Custom Dataset Processing")
print("=" * 55)

for folder_name, true_label in FOLDERS.items():
    folder_path = os.path.join(DATASET_ROOT, folder_name)
    visual_save_path = os.path.join(VISUALS_ROOT, folder_name)

    if not os.path.isdir(folder_path):
        print(f"[WARNING] Folder not found, skipping: {folder_path}")
        continue

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]

    print(f"\n[INFO] Processing '{folder_name}' ({len(image_files)} images) -> label='{true_label}'")

    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        try:
            # --- Core pipeline ---
            img  = preprocess(img_path)
            mask = segment_leaf(img)

            if np.sum(mask) == 0:
                print(f"  [SKIP] Empty mask (no green region): {fname}")
                skipped += 1
                continue

            exg     = excess_green(img, mask)
            vein    = vein_density(img, mask)
            tex     = texture_feature(img, mask)
            ndvi    = simulated_ndvi(img, mask)
            pss     = phys_stress(exg, vein, tex)
            temp    = canopy_temp(pss)

            # --- VISUAL GENERATION FOR INSTRUCTOR PROOF ---
            
            # 1. Mask Visual (Convert to RGB for stacking)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
            # 2. Vein Visual (Extract Frangi output as an image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            v = frangi(gray / 255.0)
            vein_img = (v > 0.05).astype(np.uint8) * 255
            vein_rgb = cv2.cvtColor(vein_img, cv2.COLOR_GRAY2RGB)
            
            # 3. Stress Heatmap Visual (Apply JET colormap to grayscale, then apply mask)
            heatmap_color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            heatmap_masked = cv2.bitwise_and(heatmap_color, heatmap_color, mask=mask)
            heatmap_rgb = cv2.cvtColor(heatmap_masked, cv2.COLOR_BGR2RGB) # Convert back to RGB for stacking

            # 4. Stitch Images Together (1x4 Grid)
            grid = np.hstack((img, mask_rgb, vein_rgb, heatmap_rgb))
            
            # 5. Add Labels to the Grid
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(grid, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(grid, 'Segmentation', (522, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(grid, 'Vein Detection', (1034, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(grid, f'Stress Heatmap (Temp: {temp}C)', (1546, 30), font, 1, (255, 255, 255), 2)

            # 6. Save the Grid
            out_path = os.path.join(visual_save_path, f"result_{fname}")
            cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)) # OpenCV saves in BGR

            # --- Save Data ---
            raw.append([
                fname, folder_name, true_label, exg, vein, tex, ndvi, pss, temp,
            ])
            processed += 1

        except Exception as e:
            print(f"  [ERROR] Skipping '{fname}': {e}")
            skipped += 1
            continue

print(f"\n[INFO] Done — processed: {processed}, skipped/errored: {skipped}")

# ─────────────────────────────────────────────
# BUILD DATAFRAME & NORMALIZE
# ─────────────────────────────────────────────
if processed == 0:
    print("[FATAL] No images were processed. Check DATASET_ROOT path.")
    raise SystemExit(1)

df = pd.DataFrame(raw, columns=["Image", "Folder", "True_Label", "ExG", "Vein", "Texture", "NDVI", "PSS", "Temp"])

for col in ["PSS", "NDVI", "Temp"]:
    col_std = df[col].std()
    if col_std == 0:
        df[col + "_z"] = 0.0
    else:
        df[col + "_z"] = (df[col] - df[col].mean()) / col_std

df["FinalScore"] = 0.3 * df["PSS_z"] + 0.3 * (1 - df["NDVI_z"]) + 0.4 * df["Temp_z"]

fs_min, fs_max = df["FinalScore"].min(), df["FinalScore"].max()
if fs_max - fs_min == 0:
    df["FinalScore"] = 0.5
else:
    df["FinalScore"] = (df["FinalScore"] - fs_min) / (fs_max - fs_min)

def classify(s):
    if s < 0.33: return "No Stress"
    elif s < 0.66: return "Mild Stress"
    else: return "Severe Stress"

df["Class"] = df["FinalScore"].apply(classify)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 55)
print("  PROCESSING COMPLETE")
print("=" * 55)
print(f"  Visual results saved in : {VISUALS_ROOT}/")
print(f"  Data saved in           : {OUTPUT_CSV}")