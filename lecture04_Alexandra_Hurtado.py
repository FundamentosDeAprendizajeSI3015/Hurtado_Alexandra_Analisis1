# Carga de la librería pandas y lectura del archivo CSV
import pandas as pd

df = pd.read_csv("titanic.csv")

print(df.head())


# Medidas de tendencia central---------------------------------------------------------------------
print("MEDIA")
print(df.mean(numeric_only=True))
print("\nMODA")
print(df.mode())
print("\nMEDIANA")
print(df.median(numeric_only=True))

# Medidas de dispersión---------------------------------------------------------------------------
print("\nDESVIACIÓN ESTÁNDAR (Age y Fare)")
print(df[["Age", "Fare"]].std())

print("\nVARIANZA (Age y Fare)")
print(df[["Age", "Fare"]].var())

print("\nRANGO AGE")
print(df['Age'].max() - df['Age'].min())

print("\nRANGO FARE")
print(df['Fare'].max() - df['Fare'].min())

# Percentiles e IQR
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

print("\nIQR AGE")
print(IQR)

#Graficas---------------------------------------------------------------------------------------
#Histograma edad
import matplotlib.pyplot as plt

plt.hist(df['Age'].dropna(), bins=20)
plt.title('Distribución de la Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

#Histograma tarifa
plt.hist(df['Fare'], bins=30)
plt.title('Distribución de la Tarifa')
plt.xlabel('Tarifa')
plt.ylabel('Frecuencia')
plt.show()

#BloxPlot para ver outliers
plt.boxplot(df['Age'].dropna())
plt.title('Boxplot de la Edad')
plt.ylabel('Edad')
plt.show()

#Edas vs clase
df.boxplot(column='Age', by='Pclass')
plt.title('Edad por Clase')
plt.suptitle('')
plt.xlabel('Clase')
plt.ylabel('Edad')
plt.show()

#Supervivencia vs clase 
df['Survived'].value_counts().plot(kind='bar')
plt.title('Supervivencia en el Titanic')
plt.xlabel('0 = No sobrevivió, 1 = Sobrevivió')
plt.ylabel('Cantidad de pasajeros')
plt.show()


#Transformaciones de datos----------------------------------------------------------------------
# Copia de seguridad
df_clean = df.copy()

# Ver nulos
df_clean.isnull().sum()

df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
df_clean['Embarked'] = df_clean['Embarked'].fillna(
    df_clean['Embarked'].mode()[0]
)

# Cabin se elimina (demasiados nulos)
df_clean.drop(columns=['Cabin'], inplace=True)

#Encoding 
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_clean['Sex_label'] = le.fit_transform(df_clean['Sex'])

#One-hot encoding para Embarked
df_encoded = pd.get_dummies(
    df_clean,
    columns=['Embarked', 'Pclass'],
    drop_first=True
)

#Correlación entre variables
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()

# Solo columnas numéricas (evita errores)
df_corr = df_encoded.drop(columns=['Name', 'Ticket'])

sns.heatmap(
    df_corr.corr(numeric_only=True),
    annot=False
)

plt.title("Matriz de correlación")
plt.show()

# Correlación con la variable objetivo
print(
    df_corr.corr(numeric_only=True)['Survived']
    .sort_values(ascending=False)
)

#Escalado de variables
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

cols_to_scale = ['Age', 'Fare']

df_encoded[cols_to_scale] = scaler.fit_transform(df_encoded[cols_to_scale])

#Trnsformación logaritmica para Fare
import numpy as np

df_encoded['Fare_log'] = np.log1p(df_clean['Fare'])


#Dataset Final-------------------------------------------------------------------------------------------
X = df_encoded.drop(columns=['Survived', 'Name', 'Ticket', 'Sex'])
y = df_encoded['Survived']

print("\nDATASET LIMPIO Y TRANSFORMADO (primeras filas)")
print(df_encoded.head())
print(df_encoded.tail())

"""
CONCLUSIONES DEL ANÁLISIS EXPLORATORIO DE DATOS (EDA)

En primer lugar, se puede evidenciar que la variable sexo es uno de los factores
más influyentes en la supervivencia de los pasajeros del Titanic. Tras aplicar
técnicas de codificación como One Hot Encoding, se observa una fuerte correlación
entre el sexo del pasajero y la variable objetivo Survived.

Asimismo, la clase del pasajero presenta una correlación negativa con la
supervivencia, lo que indica que los pasajeros pertenecientes a clases más bajas
tuvieron menores probabilidades de sobrevivir al desastre.

Finalmente, las técnicas de codificación de variables categóricas, junto con el
escalado de las variables numéricas, permitieron transformar el conjunto de datos
en un formato completamente numérico. Esto incrementa la calidad del dataset,
brindando un conjunto de datos limpio y correctamente transformado, lo cual mejora
la capacidad de los modelos de aprendizaje automático y aumenta la probabilidad
de obtener resultados confiables y precisos.
"""

