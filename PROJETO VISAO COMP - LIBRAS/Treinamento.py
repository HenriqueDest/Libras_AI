import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Carrega os dados
data = np.load('keypoints_data.npy')
labels = np.load('keypoints_labels.npy')

print('Formato dos dados:', data.shape)
print('Formato das labels:', labels.shape)

# Codifica as labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

print('Labels codificadas:', labels_encoded)
print('Classes:', le.classes_)

# Divide os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

print('Dados de treinamento:', X_train.shape)
print('Labels de treinamento:', y_train.shape)
print('Dados de teste:', X_test.shape)
print('Labels de teste:', y_test.shape)

# Define o número de classes
num_classes = len(le.classes_)

# Cria o modelo
model = keras.Sequential([
    layers.Input(shape=(63,)),  # 21 pontos * 3 coordenadas (x, y, z)
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compila o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Exibe a arquitetura do modelo
model.summary()

# Treina o modelo
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1
)

# Avalia o modelo nos dados de teste
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Acurácia no teste:', test_accuracy)

# Plot da acurácia
plt.plot(history.history['accuracy'], label='Acurácia de treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de validação')
plt.legend()
plt.show()

# Plot da perda
plt.plot(history.history['loss'], label='Perda de treinamento')
plt.plot(history.history['val_loss'], label='Perda de validação')
plt.legend()
plt.show()

# Salva o modelo
model.save('modelo_gestos.h5')
print('Modelo salvo como modelo_gestos.h5')
