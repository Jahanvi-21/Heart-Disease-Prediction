import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- 1. Page Configuration & Branding ---
st.set_page_config(page_title="Heart Disease Clustering Dashboard", layout="wide")

# Enhanced CSS: Fixes visibility for image_0e2134.png & adds professional polish
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 1.1rem !important;
    }
    .stMetric {
        background-color: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        padding: 20px !important;
        border-radius: 12px !important;
    }
    .main { background-color: #f8fafc; }
    h1 { color: #1e3a8a; border-bottom: 2px solid #1e3a8a; padding-bottom: 10px; }
    h2, h3 { color: #1e40af; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Data Processing Pipeline ---
@st.cache_data
def load_and_prep():
    # Loading the heart disease dataset
    df = pd.read_csv('heart_disease - heart_disease.csv')
    df_cleaned = df.dropna()
    features = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours', 'Triglyceride Level']
    X = df_cleaned[features]
    
    # Standardization for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df_cleaned, X_scaled, features

try:
    df_clean, X_scaled, feat_list = load_and_prep()

    st.title("🫀 Heart Disease Patient Segmentation Dashboard")
    st.markdown("An advanced comparison of **K-Means** (Centroid-based) and **Expectation-Maximization** (Distribution-based) methodologies.")

    # --- 3. Sidebar & Control Panel ---
    st.sidebar.header("🛠 Model Configuration")
    n_clusters = st.sidebar.select_slider("Target Segments (k)", options=[2, 3, 4, 5], value=3)
    st.sidebar.info(f"Analyzing Features: {', '.join(feat_list)}")
    run_btn = st.sidebar.button("🚀 Run Comparative Analysis", use_container_width=True)

    # Initial View: Data Preview
    if not run_btn:
        st.subheader("📋 Dataset Overview")
        st.dataframe(df_clean.head(10), use_container_width=True)
        st.info("Adjust the clusters in the sidebar and click 'Run Analysis' to see the visuals.")

    else:
        # --- 4. Model Execution ---
        # K-Means
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_scaled)
        km_sil = silhouette_score(X_scaled, km.labels_)
        
        # EM (Gaussian Mixture Model)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(X_scaled)
        gmm_labels = gmm.predict(X_scaled)
        gmm_sil = silhouette_score(X_scaled, gmm_labels)

        # --- 5. Metrics Section ---
        st.subheader("📊 Performance Analytics")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        m_col1.metric(label="K-Means Silhouette", value=f"{km_sil:.4f}", delta="Distance Based")
        m_col2.metric(label="EM (GMM) Silhouette", value=f"{gmm_sil:.4f}", delta="Distribution Based")
        m_col3.metric(label="Model Inertia (KM)", value=f"{int(km.inertia_)}")

        st.markdown("---")

        # --- 6. Visualizations (PCA) ---
        st.subheader("📍 Cluster Visualization (PCA 2D Projection)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        v_col1, v_col2 = st.columns(2)

        with v_col1:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=km.labels_, cmap='viridis', s=20, alpha=0.6)
            ax1.set_title("K-Means: Geometric Segments", fontsize=14)
            ax1.set_xlabel("PC 1")
            ax1.set_ylabel("PC 2")
            st.pyplot(fig1)

        with v_col2:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='plasma', s=20, alpha=0.6)
            ax2.set_title("EM: Probabilistic Segments", fontsize=14)
            ax2.set_xlabel("PC 1")
            ax2.set_ylabel("PC 2")
            st.pyplot(fig2)

        st.markdown("---")

        # --- 7. Technical Table ---
        st.subheader("📝 Methodological Comparison Summary")
        comparison_data = {
            "Parameter": ["Algorithm Basis", "Cluster Geometry", "Membership Type", "Silhouette Score"],
            "K-Means Clustering": ["Centroid (Euclidean)", "Spherical", "Hard Assignment", f"{km_sil:.4f}"],
            "EM (GMM)": ["Gaussian (Probability)", "Ellipsoidal", "Soft Assignment", f"{gmm_sil:.4f}"]
        }
        st.table(pd.DataFrame(comparison_data))
        
        st.success(f"Analysis complete for {n_clusters} patient segments.")

except Exception as e:
    st.error(f"Configuration Error: {e}")
