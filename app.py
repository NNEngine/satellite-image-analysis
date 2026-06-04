import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import os

st.set_page_config(
    page_title="Satellite Image Analyzer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🛰️ Satellite Image Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("Upload satellite imagery (PNG, JPG, TIFF) to extract insights, statistics, and visualizations.")

# Sidebar
st.sidebar.header("⚙️ Settings")
st.sidebar.info("This app runs on Streamlit Cloud using only pure Python packages (no GDAL required).")

# File Uploader
uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=False
)

@st.cache_data(show_spinner=False)
def load_image(uploaded_file):
    """Load image using PIL only — works everywhere including Streamlit Cloud."""
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Ensure shape is (bands, h, w)
    if len(img_array.shape) == 2:
        # Grayscale
        img_array = img_array[np.newaxis, :, :]  # (1, h, w)
        img_display = np.stack([img_array[0]] * 3, axis=-1)  # (h, w, 3)
    elif len(img_array.shape) == 3:
        # Color image (h, w, bands)
        img_display = img_array
        img_array = np.transpose(img_array, (2, 0, 1))  # (bands, h, w)
    else:
        raise ValueError("Unsupported image dimensions")
    
    return {
        'array': img_array,        # (bands, h, w)
        'display': img_display,    # (h, w, bands)
        'name': uploaded_file.name,
        'mode': image.mode
    }

def normalize_for_display(arr):
    """Normalize array to 0-255 for display."""
    arr = arr.astype(np.float32)
    min_val = np.percentile(arr, 2)
    max_val = np.percentile(arr, 98)
    arr = np.clip((arr - min_val) / (max_val - min_val + 1e-8), 0, 1)
    return (arr * 255).astype(np.uint8)

if uploaded_file is not None:
    with st.spinner("🔄 Processing image..."):
        try:
            data = load_image(uploaded_file)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()
    
    arr = data['array']  # (bands, h, w)
    display_img = data['display']
    n_bands = arr.shape[0]
    height, width = arr.shape[1], arr.shape[2]
    
    # ==================== OVERVIEW METRICS ====================
    st.markdown("---")
    st.subheader("📊 Image Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Width", f"{width:,} px")
    with col2:
        st.metric("Height", f"{height:,} px")
    with col3:
        st.metric("Bands", n_bands)
    with col4:
        st.metric("Data Type", str(arr.dtype))
    with col5:
        st.metric("Mode", data['mode'])
    
    st.info("ℹ️ Running in **PIL-only mode** — no GDAL required. For GeoTIFF metadata, use a local Python environment with rasterio installed.")
    
    # ==================== TABS ====================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🖼️ Image Viewer", 
        "📈 Band Statistics", 
        "📉 Histograms", 
        "🌿 NDVI / Indices", 
        "🔬 Spectral Profile"
    ])
    
    # -------- TAB 1: Image Viewer --------
    with tab1:
        st.subheader("Image Preview")
        
        viz_col1, viz_col2 = st.columns([3, 1])
        
        with viz_col2:
            st.markdown("**Visualization Options**")
            
            if n_bands >= 3:
                composite = st.selectbox(
                    "Color Composite",
                    ["True Color (R-G-B)", "False Color (Band 3-Band 1-Band 2)", "False Color (Band 4-Band 3-Band 1)", "Custom"]
                )
                
                if composite == "Custom":
                    r_band = st.selectbox("Red Channel", range(1, n_bands+1), index=0) - 1
                    g_band = st.selectbox("Green Channel", range(1, n_bands+1), index=min(1, n_bands-1)) - 1
                    b_band = st.selectbox("Blue Channel", range(1, n_bands+1), index=min(2, n_bands-1)) - 1
                elif composite == "True Color (R-G-B)":
                    r_band, g_band, b_band = 0, 1, 2
                elif composite == "False Color (Band 3-Band 1-Band 2)":
                    r_band, g_band, b_band = min(2, n_bands-1), 0, 1
                else:  # Band 4-Band 3-Band 1
                    r_band, g_band, b_band = min(3, n_bands-1), min(2, n_bands-1), 0
            else:
                r_band = g_band = b_band = 0
                composite = "Grayscale"
            
            enhance = st.checkbox("Auto-Enhance Contrast (2-98%)", value=True)
            show_original = st.checkbox("Show Original (no enhance)", value=False)
        
        with viz_col1:
            if n_bands >= 3 and composite != "Grayscale":
                rgb = np.stack([arr[r_band], arr[g_band], arr[b_band]], axis=-1)
                if enhance and not show_original:
                    rgb = normalize_for_display(rgb)
                fig = px.imshow(rgb, title=f"Composite: {composite}")
            else:
                band_to_show = arr[0]
                if enhance and not show_original:
                    band_to_show = normalize_for_display(band_to_show)
                fig = px.imshow(band_to_show, color_continuous_scale='gray', title="Grayscale")
            
            fig.update_layout(
                xaxis_title="Pixel Column",
                yaxis_title="Pixel Row",
                coloraxis_showscale=True,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # -------- TAB 2: Band Statistics --------
    with tab2:
        st.subheader("Per-Band Statistics")
        
        import pandas as pd
        stats_data = []
        for b in range(n_bands):
            band = arr[b]
            stats_data.append({
                'Band': f'Band {b+1}',
                'Min': np.min(band),
                'Max': np.max(band),
                'Mean': round(np.mean(band), 2),
                'Median': round(np.median(band), 2),
                'Std Dev': round(np.std(band), 2),
                'Non-Zero %': round(100 * np.count_nonzero(band) / band.size, 2)
            })
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        # Bar chart of means
        fig_means = go.Figure()
        fig_means.add_trace(go.Bar(
            x=df_stats['Band'],
            y=df_stats['Mean'],
            marker_color='#1f77b4',
            name='Mean'
        ))
        fig_means.add_trace(go.Bar(
            x=df_stats['Band'],
            y=df_stats['Std Dev'],
            marker_color='#ff7f0e',
            name='Std Dev'
        ))
        fig_means.update_layout(
            barmode='group',
            title="Mean vs Standard Deviation by Band",
            xaxis_title="Band",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig_means, use_container_width=True)
    
    # -------- TAB 3: Histograms --------
    with tab3:
        st.subheader("Pixel Distribution Histograms")
        
        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            selected_bands = st.multiselect(
                "Select bands to plot",
                [f"Band {i+1}" for i in range(n_bands)],
                default=[f"Band {i+1}" for i in range(min(3, n_bands))]
            )
        with hist_col2:
            bins = st.slider("Number of Bins", min_value=50, max_value=500, value=256)
        
        fig_hist = go.Figure()
        colors = px.colors.qualitative.Plotly
        
        for idx, band_name in enumerate(selected_bands):
            b = int(band_name.split()[1]) - 1
            band_data = arr[b].flatten()
            # Remove no-data values for histogram
            band_data = band_data[band_data > 0] if np.any(band_data == 0) else band_data
            
            fig_hist.add_trace(go.Histogram(
                x=band_data,
                name=band_name,
                opacity=0.6,
                nbinsx=bins,
                marker_color=colors[idx % len(colors)]
            ))
        
        fig_hist.update_layout(
            barmode='overlay',
            title="Pixel Intensity Distribution",
            xaxis_title="Pixel Value",
            yaxis_title="Frequency",
            height=500
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Box plot
        st.markdown("**Box Plot Comparison**")
        fig_box = go.Figure()
        for b in range(n_bands):
            fig_box.add_trace(go.Box(
                y=arr[b].flatten(),
                name=f'Band {b+1}',
                boxpoints=False
            ))
        fig_box.update_layout(
            title="Box Plot by Band",
            yaxis_title="Pixel Value",
            height=400
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # -------- TAB 4: NDVI / Indices --------
    with tab4:
        st.subheader("Vegetation & Spectral Indices")
        
        if n_bands < 2:
            st.warning("Need at least 2 bands to calculate indices.")
        else:
            idx_col1, idx_col2 = st.columns(2)
            
            with idx_col1:
                st.markdown("**NDVI (Normalized Difference Vegetation Index)**")
                red_band = st.selectbox("Red Band", range(1, n_bands+1), index=0) - 1
                nir_band = st.selectbox("NIR Band", range(1, n_bands+1), index=min(1, n_bands-1)) - 1
                
                red = arr[red_band].astype(np.float32)
                nir = arr[nir_band].astype(np.float32)
                
                # Calculate NDVI
                ndvi = np.divide(nir - red, nir + red + 1e-8, out=np.zeros_like(nir), where=(nir + red) != 0)
                ndvi = np.clip(ndvi, -1, 1)
                
                fig_ndvi = px.imshow(
                    ndvi, 
                    color_continuous_scale='RdYlGn',
                    title=f"NDVI (Red: Band {red_band+1}, NIR: Band {nir_band+1})",
                    zmin=-1, zmax=1
                )
                fig_ndvi.update_layout(height=500)
                st.plotly_chart(fig_ndvi, use_container_width=True)
                
                # NDVI stats
                ndvi_mean = np.mean(ndvi)
                ndvi_health = "Healthy 🌳" if ndvi_mean > 0.4 else "Moderate 🌿" if ndvi_mean > 0.2 else "Sparse/Stressed 🍂"
                st.metric("Average NDVI", f"{ndvi_mean:.3f}", ndvi_health)
            
            with idx_col2:
                st.markdown("**NDWI (Water Index)**")
                if n_bands >= 2:
                    green_band = st.selectbox("Green Band (for NDWI)", range(1, n_bands+1), index=min(1, n_bands-1)) - 1
                    green = arr[green_band].astype(np.float32)
                    ndwi = np.divide(green - nir, green + nir + 1e-8, out=np.zeros_like(green), where=(green + nir) != 0)
                    ndwi = np.clip(ndwi, -1, 1)
                    
                    fig_ndwi = px.imshow(
                        ndwi,
                        color_continuous_scale='Blues',
                        title=f"NDWI - Water Index (Green: Band {green_band+1}, NIR: Band {nir_band+1})",
                        zmin=-1, zmax=1
                    )
                    fig_ndwi.update_layout(height=500)
                    st.plotly_chart(fig_ndwi, use_container_width=True)
                
                # Custom Index
                st.markdown("**🧪 Custom Index Calculator**")
                num_band = st.selectbox("Numerator Band", range(1, n_bands+1), index=min(1, n_bands-1)) - 1
                den_band = st.selectbox("Denominator Band", range(1, n_bands+1), index=0) - 1
                
                num = arr[num_band].astype(np.float32)
                den = arr[den_band].astype(np.float32)
                custom_idx = np.divide(num - den, num + den + 1e-8, out=np.zeros_like(num), where=(num + den) != 0)
                
                fig_custom = px.imshow(
                    custom_idx,
                    color_continuous_scale='Viridis',
                    title=f"Custom Index: (Band {num_band+1} - Band {den_band+1}) / (Band {num_band+1} + Band {den_band+1})"
                )
                fig_custom.update_layout(height=400)
                st.plotly_chart(fig_custom, use_container_width=True)
    
    # -------- TAB 5: Spectral Profile --------
    with tab5:
        st.subheader("Spectral Profile Analysis")
        
        prof_col1, prof_col2 = st.columns([1, 3])
        
        with prof_col1:
            st.markdown("**Enter coordinates:**")
            x_coord = st.number_input("X Coordinate", min_value=0, max_value=width-1, value=width//2)
            y_coord = st.number_input("Y Coordinate", min_value=0, max_value=height-1, value=height//2)
            
            st.markdown("**Pixel Values**")
            pixel_vals = [arr[b, y_coord, x_coord] for b in range(n_bands)]
            for b, val in enumerate(pixel_vals):
                st.text(f"Band {b+1}: {val}")
        
        with prof_col2:
            # Spectral curve
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(
                x=[f"Band {i+1}" for i in range(n_bands)],
                y=pixel_vals,
                mode='lines+markers',
                line=dict(color='green', width=3),
                marker=dict(size=10),
                name=f"Pixel ({x_coord}, {y_coord})"
            ))
            
            # Add average for comparison
            avg_vals = [np.mean(arr[b]) for b in range(n_bands)]
            fig_spec.add_trace(go.Scatter(
                x=[f"Band {i+1}" for i in range(n_bands)],
                y=avg_vals,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name="Image Average"
            ))
            
            fig_spec.update_layout(
                title=f"Spectral Profile at ({x_coord}, {y_coord})",
                xaxis_title="Band",
                yaxis_title="Reflectance / DN Value",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig_spec, use_container_width=True)
        
        # 2D spectral heatmap (all pixels, sampled)
        st.markdown("**Spectral Heatmap (Sampled Pixels)**")
        sample_step = max(1, min(height, width) // 100)
        sampled = arr[:, ::sample_step, ::sample_step]  # (bands, h', w')
        n_samples = sampled.shape[1] * sampled.shape[2]
        spectral_matrix = sampled.reshape(n_bands, n_samples).T  # (samples, bands)
        
        fig_heatmap = px.imshow(
            spectral_matrix[:500],
            labels=dict(x="Band", y="Pixel Sample", color="Value"),
            title="Spectral Signatures of Sampled Pixels",
            aspect='auto'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

else:
    # Empty state
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👆 Upload a satellite image to begin analysis.")
        
        st.markdown("""
        **Supported Features:**
        - 📤 Upload PNG, JPG, or TIFF satellite imagery
        - 🖼️ Interactive viewer with true/false color composites
        - 📊 Per-band statistics (min, max, mean, std)
        - 📉 Histograms and box plots
        - 🌿 NDVI and NDWI vegetation/water indices
        - 🔬 Pixel-level spectral profiles
        - 🧪 Custom index calculator
        """)
        
        st.markdown("**Recommended Test Data:**")
        st.markdown("- [Landsat/Sentinel-2 from USGS EarthExplorer](https://earthexplorer.usgs.gov/)")
        st.markdown("- [Copernicus Open Access Hub](https://scihub.copernicus.eu/)")
