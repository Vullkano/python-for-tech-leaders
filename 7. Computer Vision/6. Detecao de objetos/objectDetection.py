# necessita de instalar previamente as bibliotecas opencv-python e ultralytics
# (esta última também instala o pytorch, que é pesado...)
#
# Testado com opencv-python 4.11 e ultralytics 8.3

import cv2
from ultralytics import YOLO

# carregar o modelo yolo26 "nano" (vai buscar automaticamente o modelo se ainda não estiver gravado)
model = YOLO("yolo26n.pt")

# caminho para o video a analisar (mudar a gosto)
video_path = "Lisbon_walk.mp4"

# inicializar um stream de video - neste caso e um ficheiro
videoStream = cv2.VideoCapture(video_path)

# definir de quantas em quantas tramas e' aplicado o detetor de objetos (FRAME_SKIP)
FRAME_SKIP = 1

# ciclo de leitura do video
frame_count = 0

while videoStream.isOpened():

    # ler uma frame de video
    read_ok, img = videoStream.read()

    # verificar se correu bem
    if not read_ok:
        print("O video chegou ao fim?")
        break

    # reduz a dimensão da imagem (passo opcional)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    # só aplica o algoritmo de FRAME_SKIP em FRAME_SKIP tramas de video
    if frame_count % FRAME_SKIP == 0:

        # obter os resultados - a biblioteca YOLO devolve uma estrutura "results" com diversos resultados
        results = model(img)

        # obter uma imagem anotada com os resultados
        img_annotated = results[0].plot()

        # mostrar
        cv2.imshow("Resultados", img_annotated)

        # termina se carregar no 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    frame_count += 1

# libertar o stream de video e fechar a janela
videoStream.release()
cv2.destroyAllWindows()