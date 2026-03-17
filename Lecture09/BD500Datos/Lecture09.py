# =========================================================
# IMPORTACIÓN DE LIBRERÍAS
# =========================================================
# Se importan librerías para:
# - manipular datos (pandas, numpy)
# - visualizar (matplotlib, seaborn)
# - aplicar modelos de clustering y métricas

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

# Estilo visual para que todos los gráficos se vean más claros
sns.set(style="whitegrid", context="notebook")


# =========================================================
# CARGA DEL DATASET
# =========================================================
print("\n" + "="*60)
print("CARGA DEL DATASET")
print("="*60)

# Se carga el dataset desde el archivo CSV
df = pd.read_csv("dataset_sintetico_FIRE_UdeA.csv")

# Se muestran dimensiones y columnas para entender la estructura
print(f"\nDimensiones: {df.shape}")
print("Columnas:", df.columns.tolist())


# =========================================================
# LIMPIEZA DE DATOS
# =========================================================
print("\n" + "="*60)
print("LIMPIEZA DE DATOS")
print("="*60)

# Se identifican valores nulos que pueden afectar el modelo
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Se guarda la variable real (label) solo para evaluación posterior
# IMPORTANTE: no se usa en el clustering
if "label" in df.columns:
    y_real = df["label"]
else:
    y_real = None

# Se seleccionan únicamente variables numéricas
# (los modelos de clustering no trabajan con texto)
X = df.select_dtypes(include=[np.number])

# Se elimina la variable label de las variables de entrada
if "label" in X.columns:
    X = X.drop(columns=["label"])

print(f"\nVariables numéricas utilizadas: {len(X.columns)}")


# =========================================================
# IMPUTACIÓN DE DATOS
# =========================================================
# En lugar de eliminar filas con valores nulos,
# se reemplazan por la mediana para no perder información

print("\nAplicando imputación con mediana...")

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("✔ Imputación completada (sin pérdida de datos)")


# =========================================================
# ESCALAMIENTO
# =========================================================
# Se normalizan las variables para que todas tengan la misma escala
# Esto es clave en clustering, ya que evita que una variable domine

print("\nEscalando datos...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("✔ Datos estandarizados (media=0, desviación=1)")


# =========================================================
# PCA (REDUCCIÓN DE DIMENSIONALIDAD)
# =========================================================
# Se reduce el número de variables manteniendo el 90% de la información
# Esto ayuda a eliminar ruido y mejorar el clustering

print("\nAplicando PCA...")

pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)

print(f"Componentes generados: {X_pca.shape[1]}")
print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2f}")


# =========================================================
# KMEANS
# =========================================================
print("\n" + "="*60)
print("KMEANS")
print("="*60)

sil_scores = []
k_range = range(2, 8)

# Se prueban diferentes valores de k para encontrar el mejor número de clusters
print("\nEvaluando diferentes valores de k:")

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_pca)

    # El silhouette mide qué tan bien separados están los clusters
    sil = silhouette_score(X_pca, labels)
    sil_scores.append(sil)

    print(f"k={k} → silhouette={sil:.4f}")

# Se selecciona automáticamente el mejor k
best_k = k_range[np.argmax(sil_scores)]

print(f"\n✔ Mejor número de clusters: {best_k}")

# Se entrena el modelo final con ese k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
labels_kmeans = kmeans.fit_predict(X_pca)

final_sil = silhouette_score(X_pca, labels_kmeans)

print(f"Silhouette final: {final_sil:.4f}")

# Interpretación básica del resultado
if final_sil < 0.2:
    print("⚠ Clusters poco definidos")
else:
    print("✔ Clusters bien separados")


# =========================================================
# DBSCAN
# =========================================================
print("\n" + "="*60)
print("DBSCAN")
print("="*60)

# Se calcula la distancia al vecino más cercano
# para ayudar a elegir el valor de eps

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_pca)
distances, _ = neighbors_fit.kneighbors(X_pca)

# Se ordenan las distancias para visualizar el "codo"
distances = np.sort(distances[:, 4])

# Gráfico para elegir eps
plt.figure(figsize=(8,5))
plt.plot(distances)
plt.title("K-distance plot (selección de eps)")
plt.xlabel("Observaciones ordenadas")
plt.ylabel("Distancia al 5to vecino")
plt.grid(alpha=0.3)
plt.show()

# Se prueban distintos valores de eps
print("\nProbando diferentes valores de eps:")

for eps in np.linspace(0.5, 3, 8):
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_pca)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"eps={eps:.2f} → clusters={n_clusters}")

# Modelo final
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_pca)


# =========================================================
# VISUALIZACIÓN
# =========================================================
print("\nVisualizando resultados...")

# Reducimos a 2D para poder graficar
pca_2d = PCA(n_components=2)
X_vis = pca_2d.fit_transform(X_scaled)

# Gráfico KMeans
plt.figure(figsize=(8,6))
plt.scatter(X_vis[:,0], X_vis[:,1], c=labels_kmeans, cmap="viridis")
plt.title("Clusters KMeans")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()

# Gráfico DBSCAN
plt.figure(figsize=(8,6))
plt.scatter(X_vis[:,0], X_vis[:,1], c=labels_dbscan, cmap="plasma")
plt.title("Clusters DBSCAN")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()


# =========================================================
# INTERPRETACIÓN
# =========================================================
# Se agregan los clusters al dataset original
df["cluster_kmeans"] = labels_kmeans

# Se analizan promedios por cluster
print("\nPromedios por cluster:")
print(df.groupby("cluster_kmeans").mean(numeric_only=True))


# =========================================================
# VALIDACIÓN
# =========================================================
# Se compara con la etiqueta real (solo como referencia)

if y_real is not None:
    print("\nEvaluación externa:")

    print("ARI KMeans:", adjusted_rand_score(y_real, labels_kmeans))
    print("ARI DBSCAN:", adjusted_rand_score(y_real, labels_dbscan))


# =========================================================
# UMAP
# =========================================================
# Visualización alternativa que puede mostrar mejor la estructura

print("\nAplicando UMAP...")

umap_model = umap.UMAP(random_state=42)
embedding = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(embedding[:,0], embedding[:,1], c=labels_kmeans, cmap="Spectral")
plt.title("UMAP - Clusters KMeans")
plt.show()
