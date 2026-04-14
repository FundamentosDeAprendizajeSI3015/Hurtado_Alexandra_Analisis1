# Clasificacion de Cancer de Mama con Redes Neuronales

Este proyecto implementa un modelo de red neuronal utilizando TensorFlow/Keras para clasificar tumores del dataset Breast Cancer Wisconsin como malignos o benignos.

Este proyecto esta basado en el ejercicio de la clase 1 trabajado previamente con el dataset Iris, adaptando la misma logica a un problema de clasificacion binaria.

## Contenido

- Carga del dataset Breast Cancer Wisconsin  
- Exploracion de datos  
- Division en entrenamiento y prueba  
- Escalado de caracteristicas  
- Construccion de una red neuronal  
- Entrenamiento del modelo  
- Evaluacion con metricas de clasificacion  
- Visualizacion del entrenamiento  

## Requisitos

Instalar las dependencias con:

```bash
pip install -r requirements.txt
```
## Ejecucion

Ejecutar el script con:

python breast_cancer_nn.py

##Dataset

Se utiliza el dataset Breast Cancer Wisconsin incluido en scikit-learn.

0: maligno
1: benigno

## Resultados

El modelo genera:

Accuracy en conjunto de prueba
Reporte de clasificacion (precision, recall, F1-score)
Matriz de confusion
Curvas de entrenamiento:
Accuracy
Loss

## Visualizacion

Se grafican las curvas de:

Accuracy de entrenamiento y validacion
Loss de entrenamiento y validacion
