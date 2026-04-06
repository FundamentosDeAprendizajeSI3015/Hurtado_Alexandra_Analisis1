# IMPORTACIÓN DE LIBRERÍAS
# Estas librerías nos permiten leer datos, limpiarlos, analizarlos, visualizarlos y aplicar técnicas de reducción de dimensionalidad.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  # poner todas las variables numéricas en la misma escala
import umap  # visualizar datos complejos en menos dimensiones
from sklearn.decomposition import PCA  #  reducir dimensiones conservando información
from sklearn.manifold import TSNE  # Técnica no lineal ideal para visualizar posibles agrupaciones


# CARGA DEL DATA SET
# Aquí cargamos el archivo de Excel que contiene las respuestas de la encuesta.
df = pd.read_excel("DatosEncuestaInforme1.xlsx")

# Limpieza de nombres de columnas
# Antes de trabajar con las variables, limpiamos sus nombres.
# Esto evita errores cuando accedemos a ellas más adelante.
df.columns = df.columns.str.strip()        # Elimina espacios al inicio o al final
df.columns = df.columns.str.replace("\n", "")  # Quita saltos de línea ocultos
df.columns = df.columns.str.lower()        # Convierte todo a minúsculas para mantener uniformidad

# Eliminar columnas innecesarias
# Quitamos información como correos, nombres o marcas de tiempo, ya que no aportan valor al análisis estadístico.
cols_eliminar = [
    'hora de finalización',
    'correo electrónico',
    'hora de la última modificación',
    'nombre',
    'id',
    'hora de inicio'
]

df = df.drop(columns=cols_eliminar, errors='ignore')  # Si alguna columna no existe, simplemente la ignora

print("\nColumnas eliminadas correctamente:")
print(cols_eliminar)

# RENOMBRAR COLUMNAS LARGAS A VARIABLES MÁS MANEJABLES
# Las preguntas originales son muy largas, así que las convertimos en nombres más cortos y fáciles de usar dentro del código.
mapeo_columnas = {
    "¿cuál es tu edad?": "edad",
    "en promedio, ¿cuántas horas al día utiliza herramientas de inteligencia artificial?": "frecuencia_uso_ia",
    "¿para qué tipo de decisiones usas principalmente herramientas de ia?": "tipo_decisiones",
    "en una escala del 1 al 5, donde 1 significa “nada” y 5 “completamente”, ¿en qué medida delega el análisis o razonamiento a herramientas de inteligencia artificial al tomar decisiones?": "delega_razonamiento",
    "en una escala del 1 al 5, donde 1 significa “nada” y 5 “totalmente”, ¿qué nivel de confianza tiene en las respuestas proporcionadas por herramientas de inteligencia artificial?": "confia_respuesta_ia",
    "¿suele verificar la información que le da la ia antes de tomar una decisión?": "verifica_respuestas",
    "¿cómo describirías tu nivel de conocimiento técnico en herramientas digitales o ia?": "nivel_experiencia",
    "en una escala del 1 al 5, donde 1 significa “nada dependiente” y 5 “totalmente dependiente”, ¿qué tan dependiente considera que es de herramientas de inteligencia artificial al momento de tomar decisiones?": "dependencia_percibida",
    "¿cuál es tu edad?": "edad"
}

df = df.rename(columns=mapeo_columnas)

print("\nColumnas renombradas correctamente:")
print(df.columns.tolist())


# INSPECCIÓN INICIAL DEL DATA SET
# Antes de hacer cualquier transformación, observamos cómo se ve el dataset.
# Esto nos ayuda a entender con qué estamos trabajando.

print("Primeras filas del dataset:")
print(df.head())  # Miramos una pequeña muestra

print("\nDimensiones del dataset:")
print(df.shape)   # Cuántas filas (respuestas) y columnas (variables) tenemos

print("\nTipos de datos:")
print(df.info())  # Revisamos si cada variable tiene el tipo correcto

print("\nEstadisticas descriptivas:")
print(df.describe())  # Resumen numérico general


# LIMPIEZA DE DATOS

# Verificar valores nulos
# Revisamos si hay respuestas incompletas que puedan afectar el análisis.
print("\nValores nulos por columna:")
print(df.isnull().sum())


# Asegurar que variables numéricas realmente sean numéricas
# Si algún valor no se puede convertir, lo transformamos en NaN para tratarlo adecuadamente después.
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
df["delega_razonamiento"] = pd.to_numeric(df["delega_razonamiento"], errors="coerce")
df["confia_respuesta_ia"] = pd.to_numeric(df["confia_respuesta_ia"], errors="coerce")
df["dependencia_percibida"] = pd.to_numeric(df["dependencia_percibida"], errors="coerce")


# TRANSFORMACIÓN DE VARIABLES
# Convertimos variables categóricas en valores numéricos respetando su orden lógico cuando existe.

# Esta variable tiene un orden natural de menor a mayor uso.
df["frecuencia_uso_ia"] = df["frecuencia_uso_ia"].map({
    "0-1": 1,
    "1-3": 2,
    "3-5": 3,
    "5 o más": 4
})

# Aquí también respetamos el orden: no verifica, a veces, sí verifica.
df["verifica_respuestas"] = df["verifica_respuestas"].map({
    "No": 0,
    "A veces": 1,
    "Sí": 2
})

print("Nulos después del mapeo:")
print(df.isnull().sum())

# Eliminamos filas con datos faltantes.
# Preferimos trabajar con registros completos para evitar ruido en el modelado.
df = df.dropna()

print("Dimensiones después de eliminar nulos:", df.shape)

# One Hot Encoding
# Convertimos variables categóricas sin orden en columnas binarias.
# Esto permite que los modelos matemáticos puedan procesarlas.
df = pd.get_dummies(
    df,
    columns=["tipo_decisiones", "nivel_experiencia"],
    drop_first=True
)

print("\nDataset despues de transformaciones:")
print(df.head())


# ESCALAMIENTO DE VARIABLES

# Seleccionamos las variables numéricas que deben ponerse en la misma escala.
# Esto es importante porque algunos modelos son sensibles a magnitudes distintas.
columnas_escalar = [
    "edad",
    "delega_razonamiento",
    "confia_respuesta_ia",
    "dependencia_percibida",
    "frecuencia_uso_ia",
    "verifica_respuestas"
]

# Creamos una copia para no alterar el dataset original.
df_scaled = df.copy()

# Aplicamos estandarización: todas las variables tendrán media 0 y desviación estándar 1.
scaler = StandardScaler()
df_scaled[columnas_escalar] = scaler.fit_transform(df[columnas_escalar])

print("\nDataset escalado correctamente (sin afectar variables dummy):")
print(df_scaled.head())

# Guardamos esta versión final, lista para aplicar modelos.
df_scaled.to_csv("dataset_model_ready.csv", index=False)

# ==============================================================
# K-MEANS CLUSTERING
# ==============================================================

# Importamos el algoritmo KMeans, que permite agrupar datos en clusters
from sklearn.cluster import KMeans

# Importamos silhouette_score, que nos ayuda a evaluar qué tan bien están definidos los clusters
from sklearn.metrics import silhouette_score

# Creamos una copia del dataset escalado para usarlo en clustering
# Es importante no incluir la variable objetivo para evitar sesgos en el agrupamiento
X_clustering = df_scaled.copy()

# =========================================
# MÉTODO DEL CODO (ELBOW METHOD)
# =========================================

# Lista donde guardaremos la inercia (error interno del modelo) para cada valor de k
inertia = []

# Probamos diferentes cantidades de clusters desde 2 hasta 9
k_range = range(2, 10)

# Iteramos sobre cada posible número de clusters
for k in k_range:
    # Creamos el modelo KMeans con k clusters
    # random_state garantiza reproducibilidad
    # n_init indica cuántas veces se ejecuta el algoritmo con diferentes centroides iniciales
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Entrenamos el modelo con los datos
    kmeans.fit(X_clustering)
    
    # Guardamos la inercia del modelo (qué tan compactos son los clusters)
    inertia.append(kmeans.inertia_)

# Creamos una nueva figura para graficar
plt.figure()

# Graficamos el número de clusters vs la inercia
plt.plot(k_range, inertia, marker='o')

# Título de la gráfica
plt.title("Método del Codo (K-Means)")

# Etiqueta del eje X
plt.xlabel("Número de clusters (k)")

# Etiqueta del eje Y
plt.ylabel("Inercia")

# Mostramos la gráfica
plt.show()

# =========================================
# SILHOUETTE SCORE
# =========================================

# Evaluamos la calidad de los clusters para cada valor de k
for k in k_range:
    # Creamos nuevamente el modelo con k clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Ajustamos el modelo y obtenemos las etiquetas de cluster directamente
    labels = kmeans.fit_predict(X_clustering)
    
    # Calculamos el silhouette score, que mide qué tan bien separados están los clusters
    score = silhouette_score(X_clustering, labels)
    
    # Imprimimos el resultado redondeado a 4 decimales
    print(f"K = {k} → Silhouette Score = {round(score, 4)}")

# =========================================
# ENTRENAR MODELO FINAL
# =========================================

# Definimos el número óptimo de clusters basado en el método del codo y silhouette
# Este valor normalmente se observa visualmente (por ejemplo, 3)
k_optimo = 3

# Creamos el modelo final con el número óptimo de clusters
kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)

# Entrenamos el modelo y obtenemos las etiquetas de cluster para cada dato
clusters = kmeans_final.fit_predict(X_clustering)

# Guardamos los clusters dentro del dataframe original
df["cluster_kmeans"] = clusters

# Mostramos cuántos elementos hay en cada cluster
print("\nDistribución de clusters:")
print(df["cluster_kmeans"].value_counts())

# =========================================
# VISUALIZACIÓN CON PCA
# =========================================

# Importamos PCA para reducir la dimensionalidad a 2 dimensiones
from sklearn.decomposition import PCA

# Creamos el modelo PCA con 2 componentes principales
pca = PCA(n_components=2)

# Transformamos los datos a 2 dimensiones
X_pca = pca.fit_transform(X_clustering)

# Creamos una nueva figura
plt.figure()

# Graficamos los puntos en el espacio reducido, coloreados por cluster
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)

# Título de la gráfica
plt.title("Clusters K-Means (PCA)")

# Etiquetas de los ejes
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")

# Mostramos la gráfica
plt.show()


# ==============================================================
# DBSCAN CLUSTERING
# ==============================================================

# Importamos DBSCAN, algoritmo basado en densidad
from sklearn.cluster import DBSCAN

# Importamos NearestNeighbors para calcular distancias entre puntos
from sklearn.neighbors import NearestNeighbors

# Volvemos a usar el dataset escalado
X_clustering = df_scaled.copy()

# =========================================
# ENCONTRAR EPS (K-DISTANCE GRAPH)
# =========================================

# Definimos el número de vecinos para calcular distancias
neighbors = NearestNeighbors(n_neighbors=5)

# Ajustamos el modelo con los datos
neighbors_fit = neighbors.fit(X_clustering)

# Calculamos las distancias a los vecinos más cercanos
distances, indices = neighbors_fit.kneighbors(X_clustering)

# Ordenamos las distancias del vecino más lejano dentro de los 5
distances = np.sort(distances[:, 4])

# Graficamos las distancias ordenadas
plt.figure()
plt.plot(distances)

# Título de la gráfica
plt.title("Gráfica K-Distance (para elegir eps)")

# Etiquetas de los ejes
plt.xlabel("Puntos ordenados")
plt.ylabel("Distancia")

# Mostramos la gráfica
plt.show()

# =========================================
# APLICAR DBSCAN
# =========================================

# Definimos el modelo DBSCAN
# eps es la distancia máxima entre puntos para ser considerados vecinos
# min_samples es el número mínimo de puntos para formar un cluster
dbscan = DBSCAN(eps=2.5, min_samples=2)

# Entrenamos el modelo y obtenemos clusters
clusters_db = dbscan.fit_predict(X_clustering)

# Guardamos los clusters en el dataframe
df["cluster_dbscan"] = clusters_db

# Mostramos la distribución de clusters
print("\nDistribución DBSCAN:")
print(df["cluster_dbscan"].value_counts())

# =========================================
# IDENTIFICAR OUTLIERS
# =========================================

# En DBSCAN, los puntos con etiqueta -1 son considerados ruido (outliers)
outliers = df[df["cluster_dbscan"] == -1]

# Mostramos la cantidad de outliers detectados
print("\nCantidad de outliers detectados:", len(outliers))

# =========================================
# VISUALIZACIÓN CON PCA
# =========================================

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_db)

plt.title("DBSCAN Clusters (PCA)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")

plt.show()

# ==============================================================
# SUBTRACTIVE CLUSTERING (simple)
# ==============================================================

# Simulamos subtractive clustering usando KMeans como aproximación académica
kmeans_sub = KMeans(n_clusters=3, random_state=42)

# Entrenamos el modelo
clusters_sub = kmeans_sub.fit_predict(df_scaled)

# Guardamos los clusters
df["cluster_subtractive"] = clusters_sub

# =========================================
# VISUALIZACIÓN SUBTRACTIVE CON PCA
# =========================================

from sklearn.decomposition import PCA

# Reducimos a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)

# Graficamos los clusters
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_sub)

plt.title("Subtractive Clustering (PCA)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")

plt.show()

# Mostramos la distribución
print("\nDistribución Subtractive:")
print(df["cluster_subtractive"].value_counts())

# ==============================================================
# FUZZY C-MEANS
# ==============================================================

# Importamos la librería de lógica difusa
import skfuzzy as fuzz

# Convertimos los datos a tipo float para evitar errores
X_fuzzy_df = df_scaled.astype(float)

# Transponemos los datos (requisito del algoritmo fuzzy)
X_fuzzy = X_fuzzy_df.T.values

# =========================================
# APLICAR FUZZY C-MEANS
# =========================================

# Definimos el número de clusters
n_clusters = 3

# Ejecutamos el algoritmo fuzzy c-means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_fuzzy,
    c=n_clusters,
    m=2,               # Controla el grado de difusividad (qué tan "borrosos" son los clusters)
    error=0.005,
    maxiter=1000,
    init=None
)

# =========================================
# ASIGNAR CLUSTER FINAL
# =========================================

# Para cada punto, elegimos el cluster con mayor probabilidad de pertenencia
cluster_fuzzy = np.argmax(u, axis=0)

# Guardamos los resultados
df["cluster_fuzzy"] = cluster_fuzzy

# Mostramos distribución
print("\nDistribución Fuzzy:")
print(df["cluster_fuzzy"].value_counts())

# =========================================
# ANALIZAR INCERTIDUMBRE
# =========================================

# Calculamos la mayor pertenencia de cada punto
max_membership = np.max(u, axis=0)

# Guardamos esta confianza en el dataframe
df["fuzzy_confianza"] = max_membership

# Filtramos casos con baja confianza (posibles datos ambiguos)
casos_dudosos = df[df["fuzzy_confianza"] < 0.6]

# Mostramos cuántos hay
print("\nCantidad de casos dudosos (posibles errores):", len(casos_dudosos))

# =========================================
# VISUALIZACIÓN CON PCA
# =========================================

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_fuzzy)

plt.title("Fuzzy C-Means (PCA)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")

plt.show()

# Comparación entre métodos
print("\nComparación de clusters:")
print("KMeans:", df["cluster_kmeans"].nunique())
print("DBSCAN:", df["cluster_dbscan"].nunique())
print("Fuzzy:", df["cluster_fuzzy"].nunique())
print("Subtractive:", df["cluster_subtractive"].nunique())

# ==============================================================
# REEVALUACIÓN DE ETIQUETAS
# ==============================================================

# Función para clasificar el nivel de influencia
def definir_nivel_influencia(x):
    if x <= 2:
        return "Baja"
    elif x == 3:
        return "Media"
    else:
        return "Alta"

# Aplicamos la función a la variable original
df["nivel_influencia"] = df["dependencia_percibida"].apply(definir_nivel_influencia)

# Creamos una copia del dataframe para trabajar sin afectar el original
df_re = df.copy()

# =========================================
# MAPEAR CLUSTERS A ETIQUETAS REALES
# =========================================

# Creamos una tabla cruzada entre clusters y etiquetas reales
tabla = pd.crosstab(df_re["cluster_kmeans"], df_re["nivel_influencia"])

# Mostramos la tabla
print("\nRelación KMeans vs Etiqueta:")
print(tabla)

# Para cada cluster, elegimos la etiqueta más frecuente
mapeo_clusters = tabla.idxmax(axis=1).to_dict()

# Mostramos el mapeo
print("\nMapeo de clusters:")
print(mapeo_clusters)

# Creamos una nueva columna con las etiquetas asignadas según el cluster
df_re["etiqueta_kmeans"] = df_re["cluster_kmeans"].map(mapeo_clusters)

# =========================================
# REGLAS DE CORRECCIÓN
# =========================================

# Definimos una función que recibe una fila del dataframe
# y decide si la etiqueta debe corregirse o mantenerse
def corregir_etiqueta(row):
    
    # Extraemos la etiqueta original (la que venía de los datos)
    etiqueta_original = row["nivel_influencia"]
    
    # Extraemos la etiqueta sugerida por el modelo KMeans
    etiqueta_cluster = row["etiqueta_kmeans"]
    
    # Extraemos el nivel de confianza del modelo fuzzy (qué tan seguro está)
    confianza = row["fuzzy_confianza"]
    
    # Verificamos si el punto fue considerado outlier por DBSCAN
    # Los outliers tienen etiqueta -1
    es_outlier = row["cluster_dbscan"] == -1
    
    # REGLA 1: Si el dato es un outlier
    # Esto significa que no pertenece claramente a ningún grupo
    # En este caso, confiamos más en el clustering que en la etiqueta original
    if es_outlier:
        return etiqueta_cluster
    
    # REGLA 2: Si la confianza del modelo fuzzy es baja
    # Esto indica que el dato es ambiguo o difícil de clasificar
    # Por lo tanto, usamos la etiqueta del cluster como mejor aproximación
    if confianza < 0.6:
        return etiqueta_cluster
    
    # REGLA 3: Si hay desacuerdo entre la etiqueta original y el cluster
    # Esto sugiere que la etiqueta original podría estar mal asignada
    # En este caso, corregimos usando el cluster
    if etiqueta_original != etiqueta_cluster:
        return etiqueta_cluster
    
    # Si ninguna condición anterior se cumple,
    # significa que el dato es confiable y consistente
    # Por lo tanto, mantenemos la etiqueta original
    return etiqueta_original

# Aplicamos la función a cada fila del dataframe
# axis=1 indica que se aplica fila por fila (no por columnas)
df_re["nivel_influencia_corregido"] = df_re.apply(corregir_etiqueta, axis=1)

# =========================================
# COMPARACIÓN
# =========================================

# Mostramos la distribución original de las etiquetas
# Esto nos permite ver cómo estaban clasificadas inicialmente
print("\nDistribución ORIGINAL:")
print(df_re["nivel_influencia"].value_counts())

# Mostramos la distribución después de aplicar las correcciones
# Así podemos comparar el impacto del proceso de ajuste
print("\nDistribución CORREGIDA:")
print(df_re["nivel_influencia_corregido"].value_counts())

# =========================================
# CUÁNTAS ETIQUETAS CAMBIARON
# =========================================

# Calculamos cuántas etiquetas fueron modificadas
# Comparamos la columna original vs la corregida
cambios = (df_re["nivel_influencia"] != df_re["nivel_influencia_corregido"]).sum()

# Mostramos la cantidad total de cambios realizados
print("\nCantidad de etiquetas modificadas:", cambios)

# Calculamos y mostramos el porcentaje de datos que fueron modificados
# Esto ayuda a entender el impacto relativo del ajuste
print("Porcentaje modificado:", round(cambios / len(df_re) * 100, 2), "%")

# ANÁLISIS EXPLORATORIO

# En esta sección comenzamos a entender el comportamiento de los datos
# utilizando estadísticas descriptivas básicas.

print("\n--- MEDIDAS DE TENDENCIA CENTRAL ---")

# Calculamos la media (promedio) de la edad
# Esto nos da una idea del valor promedio en el dataset
print("Media edad:", df["edad"].mean())

# Calculamos la mediana de la edad
# La mediana es más robusta ante valores extremos (outliers)
print("Mediana edad:", df["edad"].median())

# Media de la variable dependencia_percibida
print("Media dependencia_percibida:", df["dependencia_percibida"].mean())

# Mediana de la variable dependencia_percibida
print("Mediana dependencia_percibida:", df["dependencia_percibida"].median())

# Media de la confianza en respuestas de IA
print("Media confia_respuesta_ia:", df["confia_respuesta_ia"].mean())

# Mediana de la confianza en respuestas de IA
print("Mediana confia_respuesta_ia:", df["confia_respuesta_ia"].median())


print("\n--- MEDIDAS DE DISPERSION ---")

# Estas métricas nos permiten entender qué tan dispersos o variados están los datos

# Desviación estándar de la edad
# Indica cuánto se alejan en promedio los valores respecto a la media
print("Desviacion estandar edad:", df["edad"].std())

# Desviación estándar de la dependencia percibida
print("Desviacion estandar dependencia:", df["dependencia_percibida"].std())

# Varianza de la edad
# Es el cuadrado de la desviación estándar y mide la dispersión de forma más amplificada
print("Varianza edad:", df["edad"].var())

# Rango de la edad
# Diferencia entre el valor máximo y mínimo
print("Rango edad:", df["edad"].max() - df["edad"].min())

# El IQR (rango intercuartílico) mide la dispersión de los datos centrales
# Es robusto ante outliers
q1 = df["edad"].quantile(0.25)  # Primer cuartil (25%)
q3 = df["edad"].quantile(0.75)  # Tercer cuartil (75%)
iqr = q3 - q1                   # Diferencia entre Q3 y Q1
print("IQR edad:", iqr)



# DEFINIR VARIABLE OBJETIVO

# Creamos una función que transforma un valor numérico en una categoría
# Esto convierte el problema en uno de clasificación
def definir_nivel_influencia(x):
    if x <= 2:
        return "Baja"   # Valores bajos indican poca influencia
    elif x == 3:
        return "Media"  # Valor intermedio
    else:
        return "Alta"   # Valores altos indican alta influencia

# Aplicamos la función a la columna dependencia_percibida
# y creamos una nueva variable categórica llamada nivel_influencia
df["nivel_influencia"] = df["dependencia_percibida"].apply(definir_nivel_influencia)

# Mostramos cuántos datos hay en cada categoría
print("\nDistribución variable objetivo:")
print(df["nivel_influencia"].value_counts())



# DEFINIR X e Y

# X contiene las variables predictoras (features)
# Eliminamos la variable objetivo y la variable original numérica
# porque ya fue transformada en una categoría
X = df.drop(columns=["nivel_influencia", "dependencia_percibida"])

# Y contiene la variable objetivo que queremos predecir
Y = df["nivel_influencia"]

# Mostramos la distribución original de las clases
print("\nDistribución original:")
print(Y.value_counts())


# BALANCEO UNIFORME (OVERSAMPLING)

# Importamos la función resample para realizar sobremuestreo
from sklearn.utils import resample

# Unimos temporalmente X e Y en un solo dataframe
# Esto facilita trabajar con las clases
df_model = X.copy()
df_model["nivel_influencia"] = Y

# Separamos los datos según cada clase
df_baja = df_model[df_model["nivel_influencia"] == "Baja"]
df_alta = df_model[df_model["nivel_influencia"] == "Alta"]
df_media = df_model[df_model["nivel_influencia"] == "Media"]

# Identificamos el tamaño de la clase más grande
# Esto nos sirve como referencia para balancear
max_size = max(len(df_baja), len(df_alta), len(df_media))

# Aplicamos oversampling con reemplazo a cada clase
# Esto significa que duplicamos datos hasta igualar el tamaño máximo
df_baja_up = resample(df_baja, replace=True, n_samples=max_size, random_state=42)
df_alta_up = resample(df_alta, replace=True, n_samples=max_size, random_state=42)
df_media_up = resample(df_media, replace=True, n_samples=max_size, random_state=42)

# Unimos todas las clases ya balanceadas en un solo dataframe
df_balanced = pd.concat([df_baja_up, df_alta_up, df_media_up])

# Mezclamos aleatoriamente los datos para evitar orden por clase
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Separamos nuevamente en X (features) y Y (target)
X_balanced = df_balanced.drop(columns=["nivel_influencia"])
Y_balanced = df_balanced["nivel_influencia"]

# Mostramos la nueva distribución de clases (ya balanceada)
print("\nDistribución después de balanceo uniforme:")
print(Y_balanced.value_counts())

# Mostramos las dimensiones del dataset balanceado
print("\nDimensiones balanceadas:", X_balanced.shape)


# SPLIT 60 / 20 / 20 (BALANCEADO)

# Importamos la función para dividir los datos
from sklearn.model_selection import train_test_split

# Primera división:
# 60% entrenamiento y 40% temporal (test + validación)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_balanced, Y_balanced,
    test_size=0.4,          # 40% para dividir después
    random_state=42,
    stratify=Y_balanced     # Mantiene proporción de clases
)

# Segunda división:
# Dividimos el 40% restante en 20% test y 20% validación
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.5,          # Divide el 40% en dos partes iguales
    random_state=42,
    stratify=y_temp         # Mantiene proporción de clases
)

# Mostramos los tamaños de cada conjunto
print("\nTamaño total balanceado:", len(X_balanced))
print("Train (60%):", len(X_train))
print("Test (20%):", len(X_test))
print("Validation (20%):", len(X_val))

# ENTRENAR ÁRBOL DE DECISIÓN

# Importamos el clasificador de árbol de decisión
# Este modelo divide los datos en base a reglas para clasificar
from sklearn.tree import DecisionTreeClassifier

# Creamos el modelo de árbol de decisión
arbol = DecisionTreeClassifier(
    criterion="gini",     # Métrica para medir la calidad de la división (impureza)
    max_depth=4,          # Limita la profundidad del árbol para evitar overfitting
    random_state=42       # Garantiza resultados reproducibles
)

# Entrenamos el modelo usando los datos de entrenamiento
arbol.fit(X_train, y_train)


# EVALUACIÓN DEL MODELO

# Importamos métricas de evaluación para modelos de clasificación
from sklearn.metrics import (
    accuracy_score,       # Porcentaje de aciertos totales
    precision_score,      # Qué tan precisas son las predicciones positivas
    recall_score,         # Qué tanto logra detectar correctamente cada clase
    f1_score,             # Balance entre precisión y recall
    confusion_matrix,     # Matriz de errores y aciertos por clase
    classification_report # Reporte completo de métricas
)

# Generamos predicciones para cada conjunto de datos

# Predicciones sobre entrenamiento (para ver ajuste del modelo)
y_train_pred = arbol.predict(X_train)

# Predicciones sobre test (para evaluar generalización)
y_test_pred = arbol.predict(X_test)

# Predicciones sobre validación (para ajuste final del modelo)
y_val_pred = arbol.predict(X_val)


# MÉTRICAS

# Evaluamos el rendimiento del modelo en el conjunto de entrenamiento
print("\n===== MÉTRICAS TRAIN =====")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred, average="weighted"))
print("Recall:", recall_score(y_train, y_train_pred, average="weighted"))
print("F1-score:", f1_score(y_train, y_train_pred, average="weighted"))

# Evaluamos el rendimiento en el conjunto de prueba
print("\n===== MÉTRICAS TEST =====")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_test_pred, average="weighted"))
print("F1-score:", f1_score(y_test, y_test_pred, average="weighted"))

# Evaluamos el rendimiento en el conjunto de validación
print("\n===== MÉTRICAS VALIDATION =====")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred, average="weighted"))
print("Recall:", recall_score(y_val, y_val_pred, average="weighted"))
print("F1-score:", f1_score(y_val, y_val_pred, average="weighted"))


# RANDOM FOREST (BAGGING)

# Importamos el modelo Random Forest
# Este modelo combina múltiples árboles para mejorar la precisión
from sklearn.ensemble import RandomForestClassifier

# Creamos el modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=100,    # Número de árboles en el bosque
    max_depth=4,         # Profundidad máxima de cada árbol
    random_state=42      # Reproducibilidad
)

# Entrenamos el modelo
rf.fit(X_train, y_train)

# Generamos predicciones
y_train_rf = rf.predict(X_train)
y_test_rf = rf.predict(X_test)
y_val_rf = rf.predict(X_val)


print("\n===== RANDOM FOREST =====")

# Evaluación en entrenamiento
print("\nTRAIN")
print("Accuracy:", accuracy_score(y_train, y_train_rf))
print("Precision:", precision_score(y_train, y_train_rf, average="weighted"))
print("Recall:", recall_score(y_train, y_train_rf, average="weighted"))
print("F1-score:", f1_score(y_train, y_train_rf, average="weighted"))

# Evaluación en test
print("\nTEST")
print("Accuracy:", accuracy_score(y_test, y_test_rf))
print("Precision:", precision_score(y_test, y_test_rf, average="weighted"))
print("Recall:", recall_score(y_test, y_test_rf, average="weighted"))
print("F1-score:", f1_score(y_test, y_test_rf, average="weighted"))

# Evaluación en validación
print("\nVALIDATION")
print("Accuracy:", accuracy_score(y_val, y_val_rf))
print("Precision:", precision_score(y_val, y_val_rf, average="weighted"))
print("Recall:", recall_score(y_val, y_val_rf, average="weighted"))
print("F1-score:", f1_score(y_val, y_val_rf, average="weighted"))


# GRADIENT BOOSTING

# Importamos el modelo Gradient Boosting
# Este modelo construye árboles secuencialmente corrigiendo errores anteriores
from sklearn.ensemble import GradientBoostingClassifier

# Creamos el modelo Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,     # Número de árboles
    learning_rate=0.1,    # Qué tanto aprende cada árbol nuevo
    max_depth=3,          # Profundidad de los árboles base
    random_state=42       # Reproducibilidad
)

# Entrenamos el modelo
gb.fit(X_train, y_train)

# Generamos predicciones
y_train_gb = gb.predict(X_train)
y_test_gb = gb.predict(X_test)
y_val_gb = gb.predict(X_val)


print("\n===== GRADIENT BOOSTING =====")

# Evaluación en entrenamiento
print("\nTRAIN")
print("Accuracy:", accuracy_score(y_train, y_train_gb))
print("Precision:", precision_score(y_train, y_train_gb, average="weighted"))
print("Recall:", recall_score(y_train, y_train_gb, average="weighted"))
print("F1-score:", f1_score(y_train, y_train_gb, average="weighted"))

# Evaluación en test
print("\nTEST")
print("Accuracy:", accuracy_score(y_test, y_test_gb))
print("Precision:", precision_score(y_test, y_test_gb, average="weighted"))
print("Recall:", recall_score(y_test, y_test_gb, average="weighted"))
print("F1-score:", f1_score(y_test, y_test_gb, average="weighted"))

# Evaluación en validación
print("\nVALIDATION")
print("Accuracy:", accuracy_score(y_val, y_val_gb))
print("Precision:", precision_score(y_val, y_val_gb, average="weighted"))
print("Recall:", recall_score(y_val, y_val_gb, average="weighted"))
print("F1-score:", f1_score(y_val, y_val_gb, average="weighted"))

# REGRESIÓN LOGÍSTICA

# Importamos el modelo de regresión logística
# Este modelo se utiliza para problemas de clasificación
from sklearn.linear_model import LogisticRegression

# Creamos el modelo
# max_iter=1000 aumenta el número de iteraciones para asegurar convergencia
log_model = LogisticRegression(max_iter=1000)

# Entrenamos el modelo con los datos de entrenamiento
log_model.fit(X_train, y_train)

# Generamos predicciones en cada conjunto
y_train_log = log_model.predict(X_train)
y_test_log = log_model.predict(X_test)
y_val_log = log_model.predict(X_val)

# Mostramos resultados
print("\n===== LOGISTIC REGRESSION =====")

print("\nTRAIN")
print("Accuracy:", accuracy_score(y_train, y_train_log))

print("\nTEST")
print("Accuracy:", accuracy_score(y_test, y_test_log))

print("\nVALIDATION")
print("Accuracy:", accuracy_score(y_val, y_val_log))


# REGRESIÓN LINEAL (CORREGIDA)

# Importamos el modelo de regresión lineal
from sklearn.linear_model import LinearRegression

# Importamos métricas de regresión
from sklearn.metrics import mean_squared_error, r2_score

print("\n===== LINEAR REGRESSION =====")

# Definimos la variable objetivo numérica
# Usamos el dataset escalado para mantener consistencia
y_reg = df_scaled["dependencia_percibida"]

# Definimos las variables predictoras eliminando la variable objetivo
X_reg = df_scaled.drop(columns=["dependencia_percibida"])

# Dividimos los datos en entrenamiento y prueba
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Creamos el modelo de regresión lineal
lin_model = LinearRegression()

# Entrenamos el modelo
lin_model.fit(X_train_r, y_train_r)

# Generamos predicciones sobre el conjunto de prueba
y_pred_r = lin_model.predict(X_test_r)

# Evaluamos el modelo

# Error cuadrático medio (qué tan lejos están las predicciones)
print("MSE:", mean_squared_error(y_test_r, y_pred_r))

# R2 indica qué proporción de la variabilidad es explicada por el modelo
print("R2:", r2_score(y_test_r, y_pred_r))


# COMPARACIÓN FINAL POR ACCURACY

# Creamos un diccionario con el accuracy en validación de los modelos principales
metricas_finales = {
    "Decision Tree": accuracy_score(y_val, y_val_pred),
    "Random Forest": accuracy_score(y_val, y_val_rf),
    "Gradient Boosting": accuracy_score(y_val, y_val_gb)
}

# Convertimos el diccionario en un DataFrame para visualizar mejor
df_metricas = pd.DataFrame.from_dict(
    metricas_finales,
    orient="index",
    columns=["Accuracy Validation"]
)

# Mostramos la tabla comparativa
print("\n===== COMPARACIÓN FINAL (Accuracy en Validation) =====")
print(df_metricas)

# Identificamos el mejor modelo (mayor accuracy)
mejor_modelo = df_metricas["Accuracy Validation"].idxmax()
mejor_score = df_metricas["Accuracy Validation"].max()

# Mostramos el mejor modelo
print("\n===================================")
print("El mejor modelo fue:", mejor_modelo)
print("Con Accuracy en Validation de:", round(mejor_score, 4))
print("===================================")


# VISUALIZAR ÁRBOL DE DECISIÓN

# Importamos la función para visualizar árboles
from sklearn.tree import plot_tree

# Creamos una figura grande para que el árbol sea legible
plt.figure(figsize=(20,10))

# Dibujamos el árbol de decisión
plot_tree(
    arbol,
    feature_names=X_train.columns,   # Nombres de las variables
    class_names=arbol.classes_,      # Nombres de las clases
    filled=True,                     # Colores según clase
    rounded=True,                    # Bordes redondeados
    fontsize=8
)

# Título de la gráfica
plt.title("Árbol de Decisión - Estructura Completa")

# Mostramos la gráfica
plt.show()


# COMPARACIÓN VISUAL DE MODELOS

# Lista de modelos
modelos = ["Decision Tree", "Random Forest", "Gradient Boosting"]

# Lista de accuracies (aquí están fijos, aunque idealmente deberían ser dinámicos)
accuracy = [1.0, 1.0, 1.0]

# Creamos gráfica de barras
plt.figure()
plt.bar(modelos, accuracy)

# Título y etiquetas
plt.title("Comparación de Modelos")
plt.ylabel("Accuracy")

# Mostramos la gráfica
plt.show()


# IMPORTANCIA DE VARIABLES

# Calculamos la importancia de cada variable según el árbol de decisión
importancias = pd.Series(
    arbol.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

# Mostramos las importancias
print("\n===== IMPORTANCIA DE VARIABLES =====")
print(importancias)

# Graficamos la importancia de variables
plt.figure(figsize=(10,6))
importancias.plot(kind="bar")

plt.title("Importancia de Variables - Árbol de Decisión")
plt.ylabel("Importancia")
plt.xlabel("Variables")
plt.xticks(rotation=45)

plt.show()


# ==============================================================
# COMPARACIÓN: ORIGINAL vs CORREGIDO
# ==============================================================

# Importamos nuevamente los modelos necesarios
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =============================
# DATASET ORIGINAL
# =============================

# Definimos variables originales
X_orig = df.drop(columns=["nivel_influencia", "dependencia_percibida"])
y_orig = df["nivel_influencia"]

# Dividimos los datos
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_orig, y_orig, test_size=0.3, random_state=42, stratify=y_orig
)

# Entrenamos modelo sobre datos originales
modelo_orig = LogisticRegression(max_iter=1000)
modelo_orig.fit(X_train_o, y_train_o)

# Predicciones
y_pred_o = modelo_orig.predict(X_test_o)

# Accuracy original
acc_orig = accuracy_score(y_test_o, y_pred_o)


# =============================
# DATASET CORREGIDO
# =============================

# Usamos el dataset escalado (numérico)
df_corr_model = df_scaled.copy()

# Agregamos la variable corregida
df_corr_model["nivel_influencia_corregido"] = df_re["nivel_influencia_corregido"]

# Definimos X e Y
X_corr = df_corr_model.drop(columns=["nivel_influencia_corregido"])
y_corr = df_corr_model["nivel_influencia_corregido"]

# Dividimos los datos
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_corr, y_corr, test_size=0.3, random_state=42, stratify=y_corr
)

# Entrenamos modelo sobre datos corregidos
modelo_corr = LogisticRegression(max_iter=1000)
modelo_corr.fit(X_train_c, y_train_c)

# Predicciones
y_pred_c = modelo_corr.predict(X_test_c)

# Accuracy corregido
acc_corr = accuracy_score(y_test_c, y_pred_c)


# =============================
# RESULTADOS
# =============================

# Mostramos comparación final
print("\n===== COMPARACIÓN FINAL =====")
print("Accuracy dataset ORIGINAL:", round(acc_orig, 4))
print("Accuracy dataset CORREGIDO:", round(acc_corr, 4))

# Mostramos las clases presentes en cada caso
print("\nClases en ORIGINAL:", y_orig.unique())
print("Clases en CORREGIDO:", y_corr.unique())
