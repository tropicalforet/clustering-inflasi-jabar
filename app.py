import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Clustering Inflasi Jawa Barat", layout="wide")

st.title("📊 Dashboard Clustering Inflasi Jawa Barat")
st.write("Visualisasi hasil clustering inflasi kabupaten/kota menggunakan K-Means")

# --- Load data and perform clustering if not already done ---
try:
    df = pd.read_csv("inflasi_data_clustered.csv")
    st.success("Loaded 'inflasi_data_clustered.csv'")
except FileNotFoundError:
    st.warning("File 'inflasi_data_clustered.csv' not found. Regenerating clustering data...")
    # Re-load original data
    original_data_path = "/content/bps-od_21026_inflasi_y_on_y_brdsrkn_kabupatenkota_kelompok_peng_v2_data.csv"
    try:
        df_original = pd.read_csv(original_data_path, sep=';')
    except FileNotFoundError:
        st.error(f"Original data file '{original_data_path}' not found. Please ensure it's uploaded.")
        st.stop() # Stop Streamlit app if original data isn't there

    X = df_original.select_dtypes(include=['float64', 'int64'])
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use 6 clusters as determined previously
    kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
    df_original['Cluster'] = kmeans.fit_predict(X_scaled)
    df = df_original # Assign to df to be consistent

    # Save for future use in Streamlit
    df.to_csv('inflasi_data_clustered.csv', index=False)
    st.success("Successfully regenerated 'inflasi_data_clustered.csv'")


# --- Prepare PCA data for visualization ---
try:
    pca_df = pd.read_csv("pca_cluster_results.csv")
    st.success("Loaded 'pca_cluster_results.csv'")
except FileNotFoundError:
    st.warning("File 'pca_cluster_results.csv' not found. Regenerating PCA data...")

    # Ensure X_scaled is available for PCA
    X = df.select_dtypes(include=['float64', 'int64'])
    X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # Re-scale just to be safe

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
    pca_df['Cluster'] = df['Cluster'] # Ensure 'Cluster' column is from the current df

    # Save for future use in Streamlit
    pca_df.to_csv('pca_cluster_results.csv', index=False)
    st.success("Successfully regenerated 'pca_cluster_results.csv'")


# Sidebar
st.sidebar.header("Filter")
cluster_option = st.sidebar.multiselect(
    "Pilih Cluster",
    sorted(df['Cluster'].unique()),
    default=sorted(df['Cluster'].unique())
)

df_filtered = df[df['Cluster'].isin(cluster_option)]
pca_filtered = pca_df[pca_df['Cluster'].isin(cluster_option)]

# ===== TABEL =====
st.subheader("📋 Tabel Hasil Clustering")
st.dataframe(df_filtered)

# ===== JUMLAH DATA =====
st.subheader("📊 Jumlah Kabupaten/Kota per Cluster")
cluster_count = df_filtered['Cluster'].value_counts().sort_index()
st.bar_chart(cluster_count)

# ===== PCA SCATTER =====
st.subheader("🔍 Visualisasi PCA")
fig, ax = plt.subplots()
scatter = ax.scatter(
    pca_filtered['PC1'],
    pca_filtered['PC2'],
    c=pca_filtered['Cluster'],
    cmap='tab10'
)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
st.pyplot(fig)
