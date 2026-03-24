# =========================================================
# IMPORTACIÓN DE LIBRERÍAS
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

import umap

sns.set(style="whitegrid", context="notebook")


# =========================================================
# CARGA DEL DATASET
# =========================================================
print("\n" + "="*60)
print("CARGA DEL DATASET")
print("="*60)

df = pd.read_csv("dataset_sintetico_FIRE_UdeA.csv")

print(f"\nDimensiones: {df.shape}")
print("Columnas:", df.columns.tolist())


# =========================================================
# LIMPIEZA DE DATOS
# =========================================================
print("\n" + "="*60)
print("LIMPIEZA DE DATOS")
print("="*60)

print("\nValores nulos por columna:")
print(df.isnull().sum())

y_real = df["label"] if "label" in df.columns else None

X = df.select_dtypes(include=[np.number])

if "label" in X.columns:
    X = X.drop(columns=["label"])

print(f"\nVariables usadas: {len(X.columns)}")


# =========================================================
# IMPUTACIÓN
# =========================================================
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("\nImputación completada")


# =========================================================
# ESCALAMIENTO
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("Datos escalados")


# =========================================================
# PCA (90% varianza)
# =========================================================
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)

print(f"\nComponentes PCA: {X_pca.shape[1]}")


# =========================================================
# PCA 2D para visualización
# =========================================================
pca_2d = PCA(n_components=2)
X_vis = pca_2d.fit_transform(X_scaled)


# =========================================================
# KMEANS con 2 clusters
# =========================================================
print("\n" + "="*60)
print("KMEANS (k=2)")
print("="*60)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
labels_kmeans = kmeans.fit_predict(X_pca)

sil = silhouette_score(X_pca, labels_kmeans)
print("Silhouette:", sil)


# =========================================================
# VISUALIZACIÓN PCA 
# =========================================================
print("\nGráficas PCA...")

# Labels reales
plt.figure(figsize=(8,6))
plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="coolwarm")
plt.title("PCA - Labels reales (0 vs 1)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(label="Label")
plt.show()

# Clusters KMeans
plt.figure(figsize=(8,6))
plt.scatter(X_vis[:,0], X_vis[:,1], c=labels_kmeans, cmap="viridis")
plt.title("PCA - Clusters KMeans (k=2)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(label="Cluster")
plt.show()


# =========================================================
# INTERPRETACIÓN
# =========================================================
df["cluster_kmeans"] = labels_kmeans

print("\nPromedios por cluster:")
print(df.groupby("cluster_kmeans").mean(numeric_only=True))


# =========================================================
# VALIDACIÓN
# =========================================================
if y_real is not None:
    print("\n" + "="*60)
    print("EVALUACIÓN")
    print("="*60)

    ari = adjusted_rand_score(y_real, labels_kmeans)
    print("ARI KMeans:", ari)

    # -------------------------------
    # TABLA CRUZADA
    # -------------------------------
    print("\nDistribución porcentual (por label):")
    crosstab = pd.crosstab(y_real, labels_kmeans, normalize="index")
    print(crosstab)

    # -------------------------------
    # PORCENTAJES CLAROS
    # -------------------------------
    print("\nPorcentajes interpretados:")

    for label in [0, 1]:
        if label in crosstab.index:
            pct_c0 = crosstab.loc[label, 0]
            pct_c1 = crosstab.loc[label, 1]

            print(f"\nLabel {label}:")
            print(f"  → {pct_c0*100:.2f}% en Cluster 0")
            print(f"  → {pct_c1*100:.2f}% en Cluster 1")


# =========================================================
# UMAP
# =========================================================
print("\nUMAP...")

umap_model = umap.UMAP(random_state=42)
embedding = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(embedding[:,0], embedding[:,1], c=labels_kmeans, cmap="Spectral")
plt.title("UMAP - Clusters KMeans (k=2)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(label="Cluster")
plt.show()



# =========================================================
# DBSCAN
# =========================================================
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

print("\n" + "="*60)
print("DBSCAN")
print("="*60)

# ---------------------------------------------------------
# K-distance plot para elegir eps
# ---------------------------------------------------------
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_pca)
distances, indices = neighbors_fit.kneighbors(X_pca)

distances = np.sort(distances[:, 4])

plt.figure(figsize=(8,5))
plt.plot(distances)
plt.title("K-distance plot (elección de eps)")
plt.xlabel("Observaciones ordenadas")
plt.ylabel("Distancia al 5to vecino")
plt.grid(alpha=0.3)
plt.show()


# ---------------------------------------------------------
# Exploración automática de eps
# ---------------------------------------------------------
print("\nExploración de eps:")

for eps in np.linspace(0.5, 3, 6):
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ruido = list(labels).count(-1)

    print(f"eps={eps:.2f} → clusters={n_clusters} | ruido={ruido}")


# ---------------------------------------------------------
# DBSCAN final
# ---------------------------------------------------------
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)

n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_ruido = list(labels_dbscan).count(-1)

print(f"\nClusters encontrados: {n_clusters}")
print(f"Puntos ruido: {n_ruido}")


# =========================================================
# VISUALIZACIÓN PCA
# =========================================================
plt.figure(figsize=(8,6))

ruido = labels_dbscan == -1

plt.scatter(
    X_vis[~ruido,0],
    X_vis[~ruido,1],
    c=labels_dbscan[~ruido],
    cmap="plasma",
    s=60
)

plt.scatter(
    X_vis[ruido,0],
    X_vis[ruido,1],
    c="red",
    s=60,
    label="Ruido"
)

plt.title("DBSCAN - PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.show()


# =========================================================
# EVALUACIÓN DBSCAN
# =========================================================
if y_real is not None:

    ari_db = adjusted_rand_score(y_real, labels_dbscan)
    print("\nARI DBSCAN:", ari_db)

    print("\nDistribución porcentual DBSCAN:")
    crosstab_db = pd.crosstab(y_real, labels_dbscan, normalize="index")
    print(crosstab_db)

    print("\nPorcentajes interpretados DBSCAN:")

    for label in crosstab_db.index:
        print(f"\nLabel {label}:")
        for cluster in crosstab_db.columns:
            pct = crosstab_db.loc[label, cluster]
            print(f"  → {pct*100:.2f}% en Cluster {cluster}")
            
# =========================================================
# MATRIZ DBSCAN CON DIFERENTES EPS
# =========================================================
print("\n" + "="*60)
print("MATRIZ DBSCAN - DIFERENTES EPS")
print("="*60)

eps_values = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

fig, axes = plt.subplots(2, 3, figsize=(16,10))
axes = axes.ravel()

for i, eps in enumerate(eps_values):

    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_pca)   # clustering en alta dimensión

    ruido = labels == -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = list(labels).count(-1)

    ax = axes[i]

    # clusters
    ax.scatter(
        X_vis[~ruido,0],
        X_vis[~ruido,1],
        c=labels[~ruido],
        cmap="plasma",
        s=50
    )

    # ruido
    ax.scatter(
        X_vis[ruido,0],
        X_vis[ruido,1],
        c="red",
        s=50
    )

    ax.set_title(f"eps={eps} | clusters={n_clusters} | ruido={n_ruido}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.suptitle("Comparación DBSCAN con distintos eps", fontsize=14)
plt.tight_layout()
plt.show()
