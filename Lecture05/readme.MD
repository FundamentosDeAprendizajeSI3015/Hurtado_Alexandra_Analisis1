# Analisis del Dataset Titanic con Modelos de Regresion

Este proyecto desarrolla un analisis completo del dataset Titanic, combinando tecnicas de regresion lineal y regresion logistica para resolver problemas de prediccion continua y clasificacion.

El ejercicio incluye exploracion de datos, limpieza, modelado y evaluacion de modelos.

## Contenido

- Carga de datos desde fuente publica  
- Analisis exploratorio (EDA)  
- Limpieza y tratamiento de valores nulos  
- Codificacion de variables categoricas  
- Construccion de variables para modelado  
- Entrenamiento de modelos de regresion  
- Ajuste de hiperparametros con RandomizedSearchCV  
- Evaluacion de modelos  
- Visualizacion de resultados  

## Dataset

Se utiliza el dataset Titanic cargado desde una URL publica.

Variables principales:

- Survived: variable objetivo (clasificacion)  
- Fare: variable objetivo (regresion)  
- Age, Pclass, SibSp, Parch: variables numericas  
- Sex, Embarked: variables categoricas  

## Modelos implementados

### Regresion Lineal

Objetivo: predecir la variable continua `Fare`

Modelos utilizados:

- Ridge  
- Lasso  

Incluye:

- Transformaciones polinomiales  
- Escalado de variables  
- Busqueda de hiperparametros  

Metricas:

- R2  
- MAE  

### Regresion Logistica

Objetivo: predecir la variable binaria `Survived`

Incluye:

- Transformaciones polinomiales  
- Escalado de variables  
- Ajuste de hiperparametros  

Metricas:

- Accuracy  
- F1-score  
- Matriz de confusion  

## Visualizaciones

El proyecto genera:

- Distribucion de variables (Age, Fare)  
- Matriz de correlaciones  
- Grafico Train vs Test  
- Real vs Predicho (regresion)  
- Probabilidades de supervivencia  
- Matriz de confusion  

## Requisitos

Instalar las dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecucion

Ejecutar el script con:
```bash
python titanic_models.py
```
