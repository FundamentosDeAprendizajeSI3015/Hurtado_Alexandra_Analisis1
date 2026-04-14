# Analisis del Dataset Iris con Machine Learning

Este proyecto realiza un analisis completo del dataset Iris utilizando Python y tecnicas de Machine Learning. Incluye desde la exploracion de datos hasta el entrenamiento, evaluacion y guardado del mejor modelo.


## Contenido

- Carga y limpieza de datos  
- Analisis exploratorio (EDA)  
- Visualizacion de datos  
- Ingenieria de caracteristicas  
- Reduccion de dimensionalidad (PCA y t-SNE)  
- Entrenamiento de modelos  
- Validacion cruzada  
- Ajuste de hiperparametros  
- Evaluacion del modelo  
- Guardado del mejor modelo  


## Requisitos

Instalar las siguientes librerias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib


## Ejecucion

Ejecutar el script con:

python main.py


## Resultados

El proyecto genera:

Graficas (correlacion, pairplot, PCA, t-SNE, etc.)
Matriz de confusion
Importancia de variables (si aplica)
Reporte de metricas (accuracy, precision, recall, F1)
Modelo entrenado guardado

Todos los resultados se almacenan en la carpeta:

outputs/


## Modelos utilizados
-Regresion Logistica
-KNN
-Arbol de Decision
-Random Forest
-SVM (RBF)

Se selecciona automaticamente el mejor modelo segun su desempeño.


## Salida final

El mejor modelo se guarda en:

outputs/iris_best_model.joblib


## Dataset

Se utiliza el dataset Iris incluido en scikit-learn.
