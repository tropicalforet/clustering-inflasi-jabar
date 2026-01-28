import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Clustering Inflasi Jawa Barat",
    layout="wide"
)

st.title("📊 Dashboard Clustering Inflasi Jawa Barat")
st.write(
    "Visualisasi hasil clustering inflasi kabupaten/kota "
    "di Provinsi Jawa Barat menggunakan algoritma K-Means"
)

# ===============================
# PATH FILE
# ===============================
DATA_PATH = "bps_inflasi_jabar.csv"
CLUSTERED_PATH = "inflasi_data_clustered.csv"
PCA_PATH = "pca_cluster_results.csv"

# ===============================
# LOAD DATA MENTAH
# ===============================
@st.cache_data
def load_raw_data(path):
    return pd.read_csv(path, sep=';')

# ===============================
# PREPROCESS + CLUSTERING
# ===============================
@st.cache_data
def generate_clustering(df_raw):
    X = df_raw.select_dtypes(include=['float64', 'int64'])
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Jumlah cluster hasil evaluasi
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df_raw.copy()
    df_clustered["Cluster"] = clusters

    # PCA untuk visualisasi
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": clusters
    })

    return df_clustered, pca_df

# ===============================
# MAIN PROCESS
# ===============================
try:
    df_raw = load_raw_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"File data '{DATA_PATH}' tidak ditemukan. Pastikan file sudah di-upload ke repository GitHub.")
    st.stop()

# Coba load hasil clustering (jika sudah ada)
try:
    df = pd.read_csv(CLUSTERED_PATH)
    pca_df = pd.read_csv(PCA_PATH)
    st.success("Hasil clustering berhasil dimuat.")
except FileNotFoundError:
    st.info("Hasil clustering belum tersedia. Membuat clustering baru...")
    df, pca_df = generate_clustering(df_raw)
    df.to_csv(CLUSTERED_PATH, index=False)
    pca_df.to_csv(PCA_PATH, index=False)
    st.success("Clustering dan PCA berhasil dibuat.")

# ===============================
# SIDEBAR FILTER
# ===============================
st.sidebar.header("Filter")
cluster_option = st.sidebar.multiselect(
    "Pilih Cluster",
    sorted(df["Cluster"].unique()),
    default=sorted(df["Cluster"].unique())
)

df_filtered = df[df["Cluster"].isin(cluster_option)]
pca_filtered = pca_df[pca_df["Cluster"].isin(cluster_option)]

# ===============================
# TABEL HASIL CLUSTERING
# ===============================
st.subheader("📋 Tabel Hasil Clustering")
st.dataframe(df_filtered, use_container_width=True)

# ===============================
# GRAFIK JUMLAH DATA PER CLUSTER
# ===============================
st.subheader("📊 Jumlah Kabupaten/Kota per Cluster")
cluster_count = df_filtered["Cluster"].value_counts().sort_index()
st.bar_chart(cluster_count)

# ===============================
# VISUALISASI PCA
# ===============================
st.subheader("🔍 Visualisasi PCA")
fig, ax = plt.subplots()
ax.scatter(
    pca_filtered["PC1"],
    pca_filtered["PC2"],
    c=pca_filtered["Cluster"],
    cmap="tab10"
)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
st.pyplot(fig)

# ===============================
# CATATAN AKADEMIK
# ===============================
st.caption(
    "Clustering dilakukan secara offline menggunakan Google Colab. "
    "Aplikasi ini digunakan untuk visualisasi hasil analisis dan "
    "tidak bersifat real-time."
)

