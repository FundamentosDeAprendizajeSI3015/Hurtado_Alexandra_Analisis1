# Analisis Exploratorio de Datos (EDA) - Dataset Titanic

Este proyecto realiza un analisis exploratorio completo del dataset Titanic, incluyendo estadisticas descriptivas, visualizacion de datos y transformaciones para preparacion en Machine Learning.

El objetivo es entender las variables, identificar patrones y preparar el dataset para futuros modelos predictivos.

---

## Contenido

- Carga del dataset  
- Medidas de tendencia central (media, mediana, moda)  
- Medidas de dispersion (desviacion estandar, varianza, rango, IQR)  
- Visualizacion de datos  
- Limpieza de datos  
- Codificacion de variables categoricas  
- Analisis de correlacion  
- Escalado y transformaciones  

---

## Dataset

Se utiliza el dataset Titanic cargado desde un archivo local:

```bash
titanic.csv
```

## Variables relevantes:

-Survived: variable objetivo
-Age: edad del pasajero
-Fare: tarifa pagada
-Pclass: clase del pasajero
-Sex: sexo
-Embarked: puerto de embarque
-Analisis Exploratorio

## Se calculan:

-Media, mediana y moda
-Desviacion estandar y varianza
-Rango e IQR

Esto permite entender la distribucion y variabilidad de los datos.

## Visualizaciones

El proyecto incluye:

-Histogramas (Age y Fare)
-Boxplot para deteccion de outliers
-Edad por clase (Pclass)
-Distribucion de supervivencia
-Matriz de correlacion
-Limpieza y Transformacion

Se realizan las siguientes transformaciones:

-Imputacion de valores nulos en Age y Embarked
-Eliminacion de la variable Cabin
-Codificacion de variables:
-Label Encoding para Sex
-One Hot Encoding para Embarked y Pclass
-Escalado de variables numericas (Age, Fare)
-Transformacion logaritmica de Fare
-Dataset Final

Se construyen:

X: variables independientes
y: variable objetivo (Survived)

El dataset final queda completamente numerico y listo para modelado. 

## Ejecucion

```bash
python titanic_eda.py
```

## Conclusiones
El sexo es uno de los factores mas influyentes en la supervivencia
La clase del pasajero tiene una relacion importante con la probabilidad de sobrevivir
La limpieza y transformacion de datos mejora la calidad del dataset
Un dataset bien procesado permite obtener mejores resultados en modelos de Machine Learning
