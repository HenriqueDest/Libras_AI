import cv2
import os

# Define a pasta onde as imagens serão salvas
output_folder = 'C'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Verifica se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Não foi possível acessar a câmera.")
    exit()

# Contador para nomear as imagens
img_counter = 0

print("Pressione e segure a tecla 'Espaço' para capturar imagens. Pressione 'ESC' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar a imagem.")
        break

    # Exibe o frame na janela
    cv2.imshow("Captura de Gestos", frame)

    # Captura eventos do teclado
    key = cv2.waitKey(1) & 0xFF

    # Se a tecla espaço for pressionada, salva o frame
    if key == ord(" "):
        img_name = os.path.join(output_folder, f"gesto_{img_counter}.png")
        cv2.imwrite(img_name, frame)
        print(f"Imagem {img_name} salva!")
        img_counter += 1

    # Se a tecla ESC for pressionada, encerra o programa
    elif key == 27:  # ESC
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
