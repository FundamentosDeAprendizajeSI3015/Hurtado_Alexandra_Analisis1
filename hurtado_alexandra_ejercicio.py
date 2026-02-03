
"""
Ejercicio clase del 27 de enero. 
dataset Breast Cancer Wisconsin

"""

#  Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Reproducir 
np.random.seed(42)


#  Cargar el dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dimensiones del dataset:")
print(X.shape)

print("\nClases:")
print(y.value_counts())
# Si es 0 = maligno, 1 = benigno


#  Exploración de datos

print("\nPrimeras filas:")
print(X.head())

print("\nEstadísticas básicas:")
print(X.describe())



# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#  Construir el modelo


model = Sequential([
    Dense(
        units=16,
        activation="relu",
        input_shape=(X_train_scaled.shape[1],)
    ),

    Dense(
        units=8,
        activation="relu"
    ),

    Dense(
        units=1,
        activation="sigmoid"
    )
])


# Compilar
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Resume
print("\nResumen del modelo:")
model.summary()


# Entrenar
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=1
)


#  Evaluar
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nAccuracy en test: {accuracy:.3f}")

# Predicciones
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))


#  Curvas de entrenamiento


plt.figure(figsize=(10,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# perdida
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
