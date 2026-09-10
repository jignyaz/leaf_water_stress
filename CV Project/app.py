"""
app.py
======
Streamlit UI for the Cotton Plant Water-Stress Detector.
Traditional CV only — no deep learning.

Run with:
    streamlit run app.py
"""

import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cotton Leaf Water Stress Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Global font & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Page gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f2027 50%, #1a3a2a 100%);
    color: #e6f0e9;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(90deg, #1b4d35 0%, #2d7a4f 50%, #1b4d35 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 40px rgba(0,200,100,0.15);
    border: 1px solid #2d7a4f55;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #7effc5;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #a8d5b5;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ── Upload zone ── */
[data-testid="stFileUploadDropzone"] {
    background: #0f2218 !important;
    border: 2px dashed #2d7a4f !important;
    border-radius: 12px !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0f2218;
    border: 1px solid #2d7a4f55;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { color: #7effc5 !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #a8d5b5 !important; }

/* ── Section headers ── */
.section-head {
    font-size: 1.15rem;
    font-weight: 600;
    color: #7effc5;
    margin: 1.4rem 0 0.6rem 0;
    letter-spacing: 0.3px;
}

/* ── Verdict box ── */
.verdict-healthy {
    background: linear-gradient(90deg,#1a4d2e,#1f6b3e);
    border-left: 5px solid #4ade80;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: #4ade80;
}
.verdict-mild {
    background: linear-gradient(90deg,#4d3a00,#6b5200);
    border-left: 5px solid #facc15;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: #facc15;
}
.verdict-severe {
    background: linear-gradient(90deg,#4d1010,#6b1515);
    border-left: 5px solid #f87171;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f87171;
}

/* ── Step badges ── */
.step-badge {
    display: inline-block;
    background: #2d7a4f;
    color: #e6ffe8;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-radius: 6px;
    padding: 2px 10px;
    margin-bottom: 6px;
}

/* ── Info pill ── */
.info-pill {
    background: #0d2e1c;
    border: 1px solid #2d7a4f;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.82rem;
    color: #a8d5b5;
    display: inline-block;
    margin: 2px 4px;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🌿 Cotton Plant Water Stress Detector</p>
  <p class="hero-sub">
    Leaf Water Stress Detection in Cotton Using RGB and
Canopy Temperature
  </p>
</div>
""", unsafe_allow_html=True)

# Pipeline info pills
st.markdown("""
<div style="margin-bottom:1.2rem">
  <span class="info-pill">📐 HSV Segmentation</span>
  <span class="info-pill">🌱 Excess Green Index</span>
  <span class="info-pill">🔬 Frangi Vein Filter</span>
  <span class="info-pill">🔲 LBP Texture</span>
  <span class="info-pill">📡 Simulated NDVI</span>
  <span class="info-pill">🌡️ Canopy Temp</span>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# FILE UPLOADER
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Upload a Leaf Image</p>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a JPG or PNG cotton leaf image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#4a7a5c;font-size:0.95rem;">
        ⬆️ &nbsp; Upload an image above to start the analysis
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────────────────────
# CORE PROCESSING  (with error handling)
# ──────────────────────────────────────────────────────────────
try:
    # Decode uploaded file → numpy RGB array
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    raw        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError("Could not decode the uploaded file as an image.")
    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    # Run pipeline
    img      = preprocess(raw_rgb)
    mask     = segment_leaf(img)

    if np.sum(mask) < 500:          # fewer than 500 green pixels → not a leaf
        st.warning(
            "⚠️ **No green leaf region detected.**  "
            "Please upload a clear image of a cotton leaf against a "
            "plain or outdoor background."
        )
        st.stop()

    exg_val           = excess_green(img, mask)
    vein_val, v_overlay = vein_density(img, mask)
    tex_val           = texture_feature(img, mask)
    ndvi_val          = simulated_ndvi(img, mask)
    pss_val           = phys_stress(exg_val, vein_val, tex_val)
    temp_val          = canopy_temp(pss_val)
    heat              = stress_heatmap(img, mask)
    verdict           = classify_stress(pss_val)

except Exception as exc:
    st.error(f"❌ **Processing failed:** {exc}")
    st.info(
        "Tip: Make sure the image is a clear photograph of a cotton leaf. "
        "Very dark, blurry, or non-leaf images may fail segmentation."
    )
    st.stop()


# ──────────────────────────────────────────────────────────────
# 1 × 4  VISUAL PIPELINE GRID
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Pipeline Visualisation</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="small")

with col1:
    st.markdown('<span class="step-badge">Step 1 — Original</span>', unsafe_allow_html=True)
    st.image(img, use_container_width=True, caption="Uploaded & Resized (512×512)")

with col2:
    st.markdown('<span class="step-badge">Step 2 — Leaf Mask</span>', unsafe_allow_html=True)
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    st.image(mask_display, use_container_width=True, caption="HSV Binary Segmentation")

with col3:
    st.markdown('<span class="step-badge">Step 3 — Frangi Veins</span>', unsafe_allow_html=True)
    st.image(v_overlay, use_container_width=True, caption="Vein Network (green = veins)")

with col4:
    st.markdown('<span class="step-badge">Step 4 — Stress Heatmap</span>', unsafe_allow_html=True)
    st.image(heat, use_container_width=True, caption="ExG Heatmap (red=stressed, green=healthy)")


# ──────────────────────────────────────────────────────────────
# METRICS ROW
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Extracted Feature Metrics</p>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🌱 Excess Green (ExG)", f"{exg_val:.2f}")
m2.metric("🔬 Vein Density",        f"{vein_val:.4f}")
m3.metric("🔲 LBP Texture",         f"{tex_val:.2f}")
m4.metric("📡 Simulated NDVI",      f"{ndvi_val:.4f}")
m5.metric("⚡ PSS Score",           f"{pss_val:.4f}")


# ──────────────────────────────────────────────────────────────
# VERDICT BANNER
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Stress Classification Result</p>', unsafe_allow_html=True)

verdict_css = (
    "verdict-healthy" if "Healthy" in verdict else
    "verdict-mild"    if "Mild"    in verdict else
    "verdict-severe"
)
icon = "✅" if "Healthy" in verdict else ("⚠️" if "Mild" in verdict else "🔴")

v_col, t_col = st.columns([3, 1])
with v_col:
    st.markdown(
        f'<div class="{verdict_css}">{icon} &nbsp; {verdict}</div>',
        unsafe_allow_html=True,
    )
with t_col:
    st.metric("🌡️ Est. Canopy Temp", f"{temp_val} °C")


# ──────────────────────────────────────────────────────────────
# FEATURE BAR CHART
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Feature Importance Overview</p>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(8, 2.8))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

features = ["ExG", "Vein\nDensity", "LBP\nTexture", "Sim.\nNDVI", "PSS"]
# Normalise each to 0-1 for display (rough scale)
raw_vals   = [exg_val, vein_val * 1000, tex_val, ndvi_val * 10, pss_val]
norm_disp  = [v / max(abs(v) for v in raw_vals + [1]) for v in raw_vals]
colors     = ["#4ade80" if v >= 0 else "#f87171" for v in norm_disp]

bars = ax.bar(features, norm_disp, color=colors, edgecolor="#1a3a2a", width=0.55, zorder=3)
ax.axhline(0, color="#2d7a4f", linewidth=0.8, linestyle="--")
ax.set_ylabel("Relative magnitude", color="#a8d5b5", fontsize=9)
ax.tick_params(colors="#a8d5b5", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("#2d7a4f")
ax.grid(axis="y", color="#1a3a2a", zorder=0)

# Value labels on bars
for bar, val in zip(bars, raw_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02 * np.sign(bar.get_height() + 1e-9),
        f"{val:.3f}", ha="center", va="bottom", color="#e6f0e9", fontsize=8
    )

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#2d7a4f33;margin-top:2rem"/>
<p style="text-align:center;color:#4a7a5c;font-size:0.8rem;margin-top:0.5rem">
    Traditional CV Pipeline · cv2 · skimage · numpy · No deep learning used
</p>
""", unsafe_allow_html=True)
