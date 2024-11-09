import cv2
import mediapipe as mp
import numpy as np
import os

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Lista das pastas/classes
classes = ['A', 'B', 'C', 'D']
class_names = {'A': 'Classe A', 'B': 'Classe B', 'C': 'Classe C', 'D': 'Classe D'}

# Lista para armazenar os keypoints e as labels
data = []
labels = []

# Percorre cada classe/pasta
for class_name in classes:
    folder_path = os.path.join(class_name)
    if not os.path.exists(folder_path):
        print(f"Pasta {folder_path} não encontrada.")
        continue

    # Lista de arquivos na pasta
    images = os.listdir(folder_path)
    print(f"Processando classe {class_names[class_name]} com {len(images)} imagens.")

    # Processa cada imagem na pasta
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        # Carrega a imagem
        image = cv2.imread(img_path)
        if image is None:
            print(f"Falha ao carregar a imagem {img_path}.")
            continue

        # Converte a imagem para RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processa a imagem e extrai os keypoints
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Extrai as coordenadas (x, y, z) dos 21 pontos da mão
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])

            keypoints = np.array(keypoints).flatten()  # Transforma em um vetor 1D

            # Normaliza os keypoints (opcional, dependendo da abordagem)
            # Aqui, vamos centralizar os pontos em relação ao ponto 0 (pulso)
            wrist = keypoints[:3]
            keypoints_normalized = keypoints - np.tile(wrist, 21)

            # Adiciona aos dados
            data.append(keypoints_normalized)
            labels.append(class_names[class_name])
        else:
            print(f"Não foram detectadas mãos na imagem {img_path}.")

# Converte as listas em arrays NumPy
data = np.array(data)
labels = np.array(labels)

print("Extração concluída.")
print(f"Total de amostras: {len(data)}")

# Salva os arrays em arquivos para uso posterior
np.save('keypoints_data.npy', data)
np.save('keypoints_labels.npy', labels)

print("Dados salvos em 'keypoints_data.npy' e 'keypoints_labels.npy'.")
