# Analisis de Clustering con Dataset Sintetico y Realista

Este proyecto contiene dos implementaciones de analisis de clustering aplicadas a datasets con diferentes niveles de complejidad: uno sintetico y otro con caracteristicas mas realistas.

El objetivo es comparar el comportamiento de algoritmos de clustering y analizar la calidad de las etiquetas en los datos.

---

## Estructura del proyecto

El repositorio se divide en dos carpetas principales:

### DataSetSintetico

Contiene:

- Dataset sintetico  
- Script `Lecture10.py`  

### DataSetRealista

Contiene:

- Dataset con caracteristicas mas cercanas a un entorno real  
- Script `Lecture10.py`  

---

## Objetivo

Aplicar tecnicas de clustering no supervisado para:

- Identificar patrones en los datos  
- Evaluar la calidad de agrupaciones  
- Detectar posibles errores de etiquetado  
- Comparar resultados entre datasets  

---

## Tecnicas utilizadas

Ambos ejercicios incluyen:

- Limpieza de datos  
- Imputacion de valores faltantes  
- Escalado de variables  
- Reduccion de dimensionalidad con PCA  
- Visualizacion en 2D  

### Modelos de clustering

- KMeans  
- DBSCAN  

### Evaluacion

- Silhouette Score  
- Adjusted Rand Index (ARI)  
- Analisis de distribucion por clusters  

---

## Analisis adicional (dataset realista)

Incluye:

- Deteccion de posibles errores de clasificacion  
- Analisis por facultad (unidad)  
- Identificacion de:
  - Falsos positivos  
  - Falsos negativos  
- Comparacion entre etiquetas reales y clusters  

---

## Visualizaciones

El proyecto genera:

- Graficas PCA (labels reales vs clusters)  
- Visualizaciones con UMAP  
- Graficos de errores por categoria  
- K-distance plot para DBSCAN  
- Comparacion de diferentes valores de eps  

---

## Requisitos

Instalar las dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecucion

Ejecutar cada version de forma independiente:

```bash
cd DataSetSintetico
python Lecture10.py
```
```bash
cd ../DataSetRealista
python Lecture10.py
```

## Notas
El dataset sintetico se utiliza para entender el comportamiento base de los algoritmos
El dataset realista permite analizar problemas mas cercanos a escenarios reales
Se evita fuga de datos aplicando transformaciones solo sobre variables de entrada
