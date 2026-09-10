import cv2, os
import numpy as np
import pandas as pd
from skimage.filters import frangi
from skimage.feature import local_binary_pattern

# Redirect output to file
output_file = open("execution_log.txt", "w")

DATASET_PATH = "my_dataset/images"
AIR_TEMPERATURE = 30

# ---------------- BASIC FUNCTIONS ----------------
def preprocess(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512,512))
    return img

def segment_leaf(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (25,40,40), (100,255,255))
    return mask

def excess_green(img, mask):
    R,G,B = img[:,:,0], img[:,:,1], img[:,:,2]
    exg = 2*G - R - B
    return np.mean(exg[mask>0])

def vein_density(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    v = frangi(gray/255.0)
    veins = v > 0.05
    return np.sum(veins & (mask>0)) / (np.sum(mask>0)+1e-6)

def texture_feature(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray,8,1)
    return np.mean(lbp[mask>0])

def simulated_ndvi(img, mask):
    R = img[:,:,0].astype(float)
    G = img[:,:,1].astype(float)
    NIR = 0.5*R + 0.5*G
    ndvi = (NIR - R)/(NIR + R + 1e-6)
    return np.mean(ndvi[mask>0])

def phys_stress(exg, vein, texture):
    return 0.4*(1-exg) + 0.35*vein + 0.25*texture

def canopy_temp(pss):
    if pss < 0.3: return 31
    elif pss < 0.6: return 33
    else: return 35

# ---------------- FIRST PASS (COLLECT STATS) ----------------
raw = []
image_count = 0

try:
    for name in os.listdir(DATASET_PATH):
        try:
            img = preprocess(os.path.join(DATASET_PATH,name))
            mask = segment_leaf(img)
            if np.sum(mask)==0: continue

            exg = excess_green(img,mask)
            vein = vein_density(img,mask)
            tex = texture_feature(img,mask)
            ndvi = simulated_ndvi(img,mask)
            pss = phys_stress(exg,vein,tex)
            temp = canopy_temp(pss)

            raw.append([name, exg, vein, tex, ndvi, pss, temp])
            image_count += 1
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
    
    print(f"Processed {image_count} images")
except Exception as e:
    print(f"Error in main loop: {e}")
    import traceback
    traceback.print_exc()

df = pd.DataFrame(raw, columns=["Image","ExG","Vein","Texture","NDVI","PSS","Temp"])

# ---------------- NORMALIZATION ----------------
for col in ["PSS","NDVI","Temp"]:
    df[col+"_z"] = (df[col] - df[col].mean()) / df[col].std()

# ---------------- FINAL CALIBRATED SCORE ----------------
df["FinalScore"] = 0.3*df["PSS_z"] + 0.3*(1-df["NDVI_z"]) + 0.4*df["Temp_z"]

# Scale to 0–1
df["FinalScore"] = (df["FinalScore"] - df["FinalScore"].min()) / \
                   (df["FinalScore"].max() - df["FinalScore"].min())

def classify(s):
    if s < 0.33: return "No Stress"
    elif s < 0.66: return "Mild Stress"
    else: return "Severe Stress"

df["Class"] = df["FinalScore"].apply(classify)
df.to_csv("final_calibrated_results.csv", index=False)

print("="*50)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*50)
print("Calibrated results saved to final_calibrated_results.csv")
print(f"Total images processed: {len(df)}")
print("\nFirst 10 rows:")
print(df.head(10))

# Also write to file
output_file.write("="*50 + "\n")
output_file.write("SCRIPT COMPLETED SUCCESSFULLY\n")
output_file.write("="*50 + "\n")
output_file.write(f"Calibrated results saved to final_calibrated_results.csv\n")
output_file.write(f"Total images processed: {len(df)}\n")
output_file.write("\nFirst 10 rows:\n")
output_file.write(str(df.head(10)) + "\n")
output_file.write("\nFull Summary Statistics:\n")
output_file.write(str(df.describe()) + "\n")
output_file.close()
