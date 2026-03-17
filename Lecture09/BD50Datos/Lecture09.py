# =========================================================
# IMPORTACIÓN DE LIBRERÍAS
# =========================================================
# Librerías para:
# - Manipulación de datos
# - Visualización
# - Clustering
# - Evaluación

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

import umap

# Estilo visual global
sns.set(style="whitegrid", context="notebook")


# =========================================================
# CARGA DEL DATASET
# =========================================================
print("\n" + "="*60)
print("CARGA DEL DATASET")
print("="*60)

df = pd.read_csv("dataset_sintetico_FIRE_UdeA_realista.csv")

print(f"\nDimensiones iniciales: {df.shape}")
print("\nColumnas:")
print(df.columns.tolist())


# =========================================================
# LIMPIEZA E IMPUTACIÓN
# =========================================================
print("\n" + "="*60)
print("LIMPIEZA E IMPUTACIÓN")
print("="*60)

print("\nValores nulos por variable:")
print(df.isnull().sum())

# Guardar etiqueta real (solo para validación)
y_real = df["label"]

# Eliminar variables no útiles
print("\nEliminando variables no relevantes...")
X = df.drop(columns=["label", "unidad"], errors="ignore")

# Eliminar variable temporal
if "anio" in X.columns:
    X = X.drop(columns=["anio"])
    print("✔ 'anio' eliminada")

print(f"\nDimensiones antes de imputación: {X.shape}")

# Imputación con mediana
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print(f"Dimensiones después de imputación: {X_imputed.shape}")
print("✔ Se conservaron todos los registros")

print("\nResumen estadístico:")
print(X_imputed.describe().round(2))


# =========================================================
# ESCALAMIENTO
# =========================================================
print("\n" + "="*60)
print("ESCALAMIENTO")
print("="*60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("✔ Datos estandarizados")


# =========================================================
# PCA
# =========================================================
print("\n" + "="*60)
print("PCA - REDUCCIÓN DE DIMENSIONALIDAD")
print("="*60)

pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)

print(f"Componentes: {X_pca.shape[1]}")
print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2f}")


# =========================================================
# KMEANS
# =========================================================
print("\n" + "="*60)
print("KMEANS")
print("="*60)

sil_scores = []
k_range = range(2, 8)

print("\nEvaluación de k:")

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    sil_scores.append(sil)

    print(f"k={k} → silhouette={sil:.4f}")

best_k = k_range[np.argmax(sil_scores)]

print(f"\n✔ Mejor k: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
labels_kmeans = kmeans.fit_predict(X_pca)

final_sil = silhouette_score(X_pca, labels_kmeans)

print(f"Silhouette final: {final_sil:.4f}")

if final_sil < 0.2:
    print("⚠ Clusters débiles")
else:
    print("✔ Buena separación")


# =========================================================
# DBSCAN
# =========================================================
print("\n" + "="*60)
print("DBSCAN")
print("="*60)

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_pca)
distances, _ = neighbors_fit.kneighbors(X_pca)

distances = np.sort(distances[:, 4])

# Gráfico mejorado
plt.figure(figsize=(8,5))
plt.plot(distances, linewidth=2)
plt.title("K-distance plot", fontsize=13)
plt.xlabel("Observaciones ordenadas")
plt.ylabel("Distancia al 5to vecino")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nExplorando eps:")

for eps in np.linspace(0.5, 3, 8):
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ruido = list(labels).count(-1)

    print(f"eps={eps:.2f} → clusters={n_clusters} | ruido={ruido}")

dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)

if len(set(labels_dbscan)) > 1:
    print("✔ DBSCAN encontró clusters")
else:
    print("⚠ No encontró estructura clara")


# =========================================================
# VISUALIZACIÓN
# =========================================================
print("\n" + "="*60)
print("VISUALIZACIÓN")
print("="*60)

pca_2d = PCA(n_components=2)
X_vis = pca_2d.fit_transform(X_scaled)

# KMeans
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    X_vis[:,0], X_vis[:,1],
    c=labels_kmeans,
    cmap="viridis",
    s=60,
    alpha=0.8,
    edgecolor="black"
)
plt.title("Clusters KMeans (PCA)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# DBSCAN
plt.figure(figsize=(8,6))

ruido = labels_dbscan == -1

plt.scatter(
    X_vis[~ruido,0], X_vis[~ruido,1],
    c=labels_dbscan[~ruido],
    cmap="plasma",
    s=60,
    alpha=0.8,
    edgecolor="black",
    label="Clusters"
)

plt.scatter(
    X_vis[ruido,0], X_vis[ruido,1],
    c="red",
    s=60,
    label="Ruido"
)

plt.title("DBSCAN (detección de ruido)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# INTERPRETACIÓN
# =========================================================
print("\n" + "="*60)
print("INTERPRETACIÓN")
print("="*60)

df["cluster_kmeans"] = labels_kmeans

print("\nPromedios por cluster:")
print(df.groupby("cluster_kmeans").mean(numeric_only=True).round(2))


# =========================================================
# VALIDACIÓN
# =========================================================
print("\n" + "="*60)
print("EVALUACIÓN")
print("="*60)

ari_k = adjusted_rand_score(y_real, labels_kmeans)
ari_d = adjusted_rand_score(y_real, labels_dbscan)

print(f"ARI KMeans: {ari_k:.4f}")
print(f"ARI DBSCAN: {ari_d:.4f}")

print("\nInterpretación:")
print("- 1 → perfecta coincidencia")
print("- 0 → sin relación")
print("- negativo → desacuerdo")


# =========================================================
# UMAP
# =========================================================
print("\n" + "="*60)
print("UMAP")
print("="*60)

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    embedding[:,0], embedding[:,1],
    c=labels_kmeans,
    cmap="Spectral",
    s=60,
    alpha=0.85,
    edgecolor="black"
)

plt.title("UMAP - Clusters KMeans")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
