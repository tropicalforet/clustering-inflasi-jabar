import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Peta Kelompok Inflasi Jawa Barat",
    layout="wide"
)

# ===============================
# HEADER
# ===============================
st.title("📊 Peta Kelompok Inflasi Kabupaten/Kota di Jawa Barat")
st.markdown(
    """
    Aplikasi ini menampilkan hasil **pengelompokan (clustering)** wilayah di Jawa Barat  
    berdasarkan **pola inflasi kelompok pengeluaran**.

    👉 Tujuannya adalah membantu memahami **perbedaan karakteristik inflasi antar wilayah**  
    secara **visual dan mudah dipahami**.
    """
)

st.divider()

# ===============================
# PATH FILE
# ===============================
DATA_PATH = "bps_inflasi_jabar.csv"
CLUSTERED_PATH = "inflasi_data_clustered.csv"
PCA_PATH = "pca_cluster_results.csv"

# ===============================
# LOAD & PROCESS DATA
# ===============================
@st.cache_data
def load_raw_data(path):
    return pd.read_csv(path, sep=';')

@st.cache_data
def generate_clustering(df_raw):
    X = df_raw.select_dtypes(include=['float64', 'int64'])
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df_raw.copy()
    df_clustered["Cluster"] = clusters

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": clusters
    })

    return df_clustered, pca_df

# ===============================
# MAIN LOAD
# ===============================
try:
    df_raw = load_raw_data(DATA_PATH)
except FileNotFoundError:
    st.error("❌ Data inflasi tidak ditemukan. Pastikan file CSV sudah di-upload.")
    st.stop()

try:
    df = pd.read_csv(CLUSTERED_PATH)
    pca_df = pd.read_csv(PCA_PATH)
except FileNotFoundError:
    df, pca_df = generate_clustering(df_raw)
    df.to_csv(CLUSTERED_PATH, index=False)
    pca_df.to_csv(PCA_PATH, index=False)

# ===============================
# RINGKASAN UTAMA
# ===============================
st.subheader("📌 Ringkasan Hasil")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Wilayah", df.shape[0])

with col2:
    st.metric("Jumlah Kelompok (Cluster)", df["Cluster"].nunique())

with col3:
    st.metric("Metode Analisis", "K-Means")

st.info(
    "Wilayah dikelompokkan ke dalam **6 cluster**, di mana setiap cluster "
    "memiliki karakteristik inflasi yang relatif mirip."
)

st.divider()

# ===============================
# SIDEBAR FILTER
# ===============================
st.sidebar.header("🔎 Filter Tampilan")
cluster_option = st.sidebar.multiselect(
    "Pilih Kelompok Wilayah (Cluster)",
    sorted(df["Cluster"].unique()),
    default=sorted(df["Cluster"].unique())
)

df_filtered = df[df["Cluster"].isin(cluster_option)]
pca_filtered = pca_df[pca_df["Cluster"].isin(cluster_option)]

# ===============================
# GRAFIK JUMLAH WILAYAH
# ===============================
st.subheader("📊 Jumlah Wilayah di Setiap Kelompok")
st.markdown(
    "Grafik berikut menunjukkan **berapa banyak kabupaten/kota** "
    "yang masuk ke masing-masing kelompok inflasi."
)

cluster_count = df_filtered["Cluster"].value_counts().sort_index()
st.bar_chart(cluster_count)

st.divider()

# ===============================
# PCA VISUAL
# ===============================
st.subheader("🗺️ Peta Sebaran Kelompok Wilayah")
st.markdown(
    """
    Visualisasi ini membantu melihat **pola kedekatan antar wilayah**.
    - Titik yang **berdekatan** → karakteristik inflasi mirip  
    - Warna yang sama → berada dalam **kelompok inflasi yang sama**
    """
)

fig, ax = plt.subplots()
scatter = ax.scatter(
    pca_filtered["PC1"],
    pca_filtered["PC2"],
    c=pca_filtered["Cluster"],
    cmap="tab10"
)
ax.set_xlabel("Dimensi Pola Inflasi (1)")
ax.set_ylabel("Dimensi Pola Inflasi (2)")
st.pyplot(fig)

st.divider()

# ===============================
# TABEL DETAIL
# ===============================
st.subheader("📋 Daftar Kabupaten/Kota dan Kelompok Inflasi")
st.markdown(
    "Tabel berikut menampilkan **detail hasil pengelompokan** "
    "untuk setiap wilayah."
)

st.dataframe(df_filtered, use_container_width=True)

# ===============================
# FOOTNOTE
# ===============================
st.caption(
    "Catatan: Pengelompokan dilakukan menggunakan algoritma K-Means. "
    "Visualisasi PCA hanya digunakan untuk membantu pemahaman dan "
    "tidak memengaruhi hasil clustering."
)
