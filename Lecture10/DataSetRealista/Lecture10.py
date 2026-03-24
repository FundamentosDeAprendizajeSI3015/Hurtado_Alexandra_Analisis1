# =========================================================
# IMPORTACIÓN DE LIBRERÍAS
# =========================================================

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

# Guardar label (solo evaluación)
y_real = df["label"]

# quitar label
X = df.drop(columns=["label", "unidad"], errors="ignore")

if "anio" in X.columns:
    X = X.drop(columns=["anio"])

print(f"\nDimensiones antes de imputación: {X.shape}")

# Imputación
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("✔ Imputación completada")


# =========================================================
# ESCALAMIENTO
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("✔ Datos estandarizados")


# =========================================================
# PCA (ALTA DIMENSIÓN PARA CLUSTERING)
# =========================================================
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)

print(f"\nComponentes PCA: {X_pca.shape[1]}")

# PCA 2D SOLO PARA VISUALIZAR
pca_2d = PCA(n_components=2)
X_vis = pca_2d.fit_transform(X_scaled)


# =========================================================
# KMEANS
# =========================================================
print("\n" + "="*60)
print("KMEANS (k=2)")
print("="*60)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
labels_kmeans = kmeans.fit_predict(X_pca)

sil = silhouette_score(X_pca, labels_kmeans)
print(f"Silhouette: {sil:.4f}")


# =========================================================
# DETECTAR POSIBLES ERRORES DE ETIQUETADO
# =========================================================
print("\n" + "="*60)
print("POSIBLES ERRORES POR FACULTAD")
print("="*60)

# guardar cluster en dataframe
df["cluster_kmeans"] = labels_kmeans

# crear columna error
df["posible_error"] = df["label"] != df["cluster_kmeans"]

# contar errores por facultad
errores_por_facultad = (
    df.groupby("unidad")["posible_error"]
    .mean()
    .sort_values(ascending=False)
)

print("\nPorcentaje de posibles errores por facultad:")
print((errores_por_facultad * 100).round(2))


plt.figure(figsize=(10,5))

(errores_por_facultad*100).plot(kind="bar")

plt.title("Porcentaje de posible mala clasificación por facultad")
plt.ylabel("% posibles errores")
plt.xlabel("Facultad")
plt.xticks(rotation=45)
plt.show()

print("\nFilas con posible mala clasificación:")
print(
    df[df["posible_error"]]
    [["unidad", "label", "cluster_kmeans"]]
    .head(20)
)


# =========================================================
# FALSOS POSITIVOS POR FACULTAD
# =========================================================
print("\n" + "="*60)
print("FALSOS POSITIVOS POR FACULTAD")
print("="*60)

# falso positivo: label=1 pero cluster=0
df["falso_positivo"] = (df["label"] == 1) & (df["cluster_kmeans"] == 0)

# porcentaje por facultad
fp_por_facultad = (
    df.groupby("unidad")["falso_positivo"]
    .mean()
    .sort_values(ascending=False)
)

print("\nPorcentaje de falsos positivos por facultad:")
print((fp_por_facultad*100).round(2))

plt.figure(figsize=(10,5))

(fp_por_facultad*100).plot(kind="bar")

plt.title("Falsos positivos por facultad")
plt.ylabel("% Falsos positivos")
plt.xlabel("Facultad")
plt.xticks(rotation=45)
plt.show()

peor_fp = fp_por_facultad.idxmax()
valor_fp = fp_por_facultad.max()

print(f"\nFacultad con más falsos positivos: {peor_fp}")
print(f"Porcentaje: {valor_fp*100:.2f}%")


# =========================================================
# FALSOS NEGATIVOS
# =========================================================
df["falso_negativo"] = (df["label"] == 0) & (df["cluster_kmeans"] == 1)

# porcentaje por facultad
fn_por_facultad = (
    df.groupby("unidad")["falso_negativo"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,5))

(fn_por_facultad * 100).plot(kind="bar")

plt.title("Falsos negativos por facultad")
plt.ylabel("% Falsos negativos")
plt.xlabel("Facultad")
plt.xticks(rotation=45)
plt.show()

print("Total falsos negativos:", df["falso_negativo"].sum())

print(
    df[df["falso_negativo"]]
    [["unidad","label","cluster_kmeans"]]
    
)

df["falso_positivo"] = (df["label"] == 1) & (df["cluster_kmeans"] == 0)

comparacion = pd.DataFrame({
    "Falsos positivos": df.groupby("unidad")["falso_positivo"].mean(),
    "Falsos negativos": df.groupby("unidad")["falso_negativo"].mean()
}) * 100

comparacion.plot(kind="bar", figsize=(12,6))

plt.title("Falsos positivos vs falsos negativos por facultad")
plt.ylabel("Porcentaje")
plt.xlabel("Facultad")
plt.xticks(rotation=45)
plt.show()

# =========================================================
# VISUALIZACIÓN PCA
# =========================================================
plt.figure(figsize=(8,6))
plt.scatter(X_vis[:,0], X_vis[:,1], c=y_real, cmap="coolwarm")
plt.title("PCA - Labels reales")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Label")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(X_vis[:,0], X_vis[:,1], c=labels_kmeans, cmap="viridis")
plt.title("PCA - Clusters KMeans")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Cluster")
plt.show()


# =========================================================
# EVALUACIÓN KMEANS
# =========================================================
ari_k = adjusted_rand_score(y_real, labels_kmeans)
print(f"\nARI KMeans: {ari_k:.4f}")

crosstab = pd.crosstab(y_real, labels_kmeans, normalize="index")
print("\nDistribución porcentual:")
print(crosstab)


# =========================================================
# UMAP
# =========================================================
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(embedding[:,0], embedding[:,1], c=labels_kmeans, cmap="Spectral")
plt.title("UMAP - KMeans")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(label="Cluster")
plt.show()


# =========================================================
# DBSCAN - K DISTANCE
# =========================================================
print("\n" + "="*60)
print("DBSCAN")
print("="*60)

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_pca)
distances, indices = neighbors_fit.kneighbors(X_pca)

distances = np.sort(distances[:, 4])

plt.figure(figsize=(8,5))
plt.plot(distances)
plt.title("K-distance plot")
plt.xlabel("Observaciones")
plt.ylabel("Distancia")
plt.show()


# =========================================================
# EXPLORACIÓN EPS
# =========================================================
print("\nExploración eps")

for eps in np.linspace(0.5, 3, 6):
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ruido = list(labels).count(-1)

    print(f"eps={eps:.2f} → clusters={n_clusters} | ruido={ruido}")


# =========================================================
# DBSCAN FINAL
# =========================================================
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)

ruido = labels_dbscan == -1

plt.figure(figsize=(8,6))

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
    s=60
)

plt.title("DBSCAN final")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# =========================================================
# MATRIZ DBSCAN DIFERENTES EPS
# =========================================================
print("\nMatriz DBSCAN")

eps_values = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

fig, axes = plt.subplots(2, 3, figsize=(16,10))
axes = axes.ravel()

for i, eps in enumerate(eps_values):

    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_pca)

    ruido = labels == -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = list(labels).count(-1)

    ax = axes[i]

    ax.scatter(
        X_vis[~ruido,0],
        X_vis[~ruido,1],
        c=labels[~ruido],
        cmap="plasma",
        s=50
    )

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
