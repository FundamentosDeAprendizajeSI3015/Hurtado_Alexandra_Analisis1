# Actividad de Clase: Modelos Basados en Árboles, Clustering y Mejora de Calidad de Datos

## Contexto

Esta actividad se desarrolla como una **extensión del pipeline de preprocesamiento previamente construido**, el cual incluía limpieza de datos, transformación de variables y preparación del dataset para modelado.

Sobre este pipeline base, se incorporaron nuevas etapas con el objetivo de **enriquecer el análisis y mejorar la calidad de los datos**, integrando técnicas de aprendizaje no supervisado y validación de etiquetas.

---

## Objetivo

Aplicar técnicas avanzadas de análisis de datos y machine learning para predecir el **nivel de influencia del uso de herramientas de Inteligencia Artificial**, no solo entrenando modelos, sino también **mejorando la calidad del dataset mediante clustering y corrección de etiquetas**.

---

## Pipeline Utilizado

Se reutilizó y amplió el pipeline previamente construido, incluyendo:

1. Limpieza y estandarización de nombres de columnas  
2. Eliminación de variables irrelevantes  
3. Conversión de variables numéricas  
4. Codificación ordinal de variables con orden natural  
5. One Hot Encoding para variables categóricas nominales  
6. Estandarización de variables numéricas  
7. Construcción de la variable objetivo (`nivel_influencia`)
8. Análisis exploratorio de datos (EDA)  

### Nuevas etapas incorporadas

8. Aplicación de múltiples algoritmos de clustering  
9. Evaluación de estructura de datos (Elbow Method y Silhouette Score)  
10. Detección de outliers (DBSCAN)  
11. Análisis de incertidumbre (Fuzzy C-Means)  
12. Sistema de corrección de etiquetas basado en reglas  
13. Comparación entre dataset original vs dataset corregido  

---


## Clustering y Análisis No Supervisado

Se implementaron múltiples técnicas de clustering para descubrir patrones ocultos:

- **K-Means:** Identificación de grupos principales  
- **DBSCAN:** Detección de outliers  
- **Fuzzy C-Means:** Análisis de pertenencia difusa e incertidumbre  
- **Subtractive Clustering:** Aproximación académica adicional  

Estos métodos permitieron entender mejor la estructura interna de los datos y detectar posibles inconsistencias.

---

## Corrección de Etiquetas (Data Quality Improvement)

Se diseñó un sistema de corrección basado en reglas utilizando la información obtenida del clustering:

- Si un dato es outlier → se corrige usando el cluster  
- Si la confianza fuzzy es baja → se corrige  
- Si existe inconsistencia entre etiqueta original y cluster → se corrige  

Esto permitió mejorar la calidad de la variable objetivo antes del modelado.

---

## Balanceo de Clases

Se aplicó oversampling uniforme para equilibrar las clases:

- Muestreo con reemplazo  
- Igualación al tamaño de la clase mayoritaria  

Clases balanceadas:
- Baja  
- Media  
- Alta  

---

## División del Dataset

El dataset balanceado fue dividido de forma estratificada en:

- Entrenamiento (60%)  
- Prueba (20%)  
- Validación (20%)  

Esto garantiza una evaluación más confiable del modelo.

---

## Modelos Implementados

Se entrenaron y compararon los siguientes modelos:

- Decision Tree  
- Random Forest  
- Gradient Boosting  
- Logistic Regression  

Adicionalmente, se implementó un modelo de **regresión lineal** para analizar la variable numérica original.

---

## Métricas de Evaluación

Se utilizaron múltiples métricas para evaluar el desempeño:

- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1-score (weighted)  
- Mean Squared Error (para regresión)  
- R² Score  

El criterio principal de comparación fue el **Accuracy en el conjunto de validación**.

---

## Comparación: Dataset Original vs Corregido

Se realizó un experimento clave:

- Entrenar modelos con el dataset original  
- Entrenar modelos con el dataset corregido  

**Resultado:**  
El dataset corregido mostró **mejor desempeño**, evidenciando la importancia de la calidad de los datos en machine learning.

---

## Interpretabilidad del Modelo

Se aplicaron técnicas de interpretabilidad:

- Visualización del árbol de decisión  
- Importancia de variables (`feature_importances_`)  

Esto permitió entender qué variables influyen más en la predicción.

---

## Resultados

Los modelos ensemble (Random Forest y Gradient Boosting) presentaron mejor desempeño en comparación con modelos individuales.

Se identificaron variables clave que influyen en el nivel de influencia, lo que aporta valor interpretativo al modelo.

---

## Tecnologías Utilizadas

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Scikit-fuzzy  
- UMAP-learn  
- SciPy  

---

## Conclusiones

Este proyecto demostró que el rendimiento de un modelo no depende únicamente del algoritmo, sino principalmente de la calidad de los datos.

La integración de técnicas de clustering permitió no solo identificar patrones, sino también detectar inconsistencias y mejorar las etiquetas del dataset.

El uso de métodos como DBSCAN y Fuzzy C-Means aportó una capa adicional de análisis, permitiendo identificar outliers y medir la incertidumbre en la clasificación.

Asimismo, se evidenció que los modelos ensemble ofrecen mejor capacidad de generalización frente a modelos simples.

Finalmente, la comparación entre dataset original y corregido confirmó que **mejorar la calidad de los datos tiene un impacto directo y significativo en el desempeño del modelo**, consolidando la importancia del preprocesamiento en proyectos de machine learning.
