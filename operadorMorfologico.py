# pip install opencv-python
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

# Lista das imagens a processar
nomes_imgs = ['GIRAFA.jpg', 'Satelite.jpg', 'Aviao.jpg']

for nome_img in nomes_imgs:
    # Parâmetros editáveis para cada imagem
    if nome_img == 'GIRAFA.jpg':
        tamanho_blur = 5
        offset_thresh = 90
        kernel_size = 10
    elif nome_img == 'Satelite.jpg':
        tamanho_blur = 1
        offset_thresh = 80
        kernel_size = 6
    elif nome_img == 'Aviao.jpg':
        tamanho_blur = 1
        offset_thresh = 50
        kernel_size = 7

    # Lê e converte imagem
    img = cv2.imread(f'./Images/{nome_img}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Blur
    img_blur = cv2.blur(img, (tamanho_blur, tamanho_blur))

    # Cinza
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)

    # Threshold binário invertido
    a = img_gray.max()
    _, thresh = cv2.threshold(img_gray, a / 2 + offset_thresh, a, cv2.THRESH_BINARY_INV)

    # Kernel para operações morfológicas
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Operações morfológicas
    img_dilate = cv2.dilate(thresh, kernel, iterations=1)
    img_erode = cv2.erode(thresh, kernel, iterations=1)
    img_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    img_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    img_grad = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    img_tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)
    img_blackhat = cv2.morphologyEx(thresh, cv2.MORPH_BLACKHAT, kernel)

    # Lista de imagens e títulos
    imagens = [
        img, img_blur, img_gray, thresh,
        img_erode, img_dilate, img_open,
        img_close, img_grad, img_tophat, img_blackhat
    ]

    titles = ["Imagem original", "Imagem com blur", "Escala de cinza", "Threshold binário",
              "Erosão", "Dilatação", "Abertura morfológica", "Fechamento morfológico",
              "Gradiente morfológico", "Top Hat", "Black Hat"]

    # Grid automático
    formatoX = math.ceil(len(imagens) ** 0.5)
    formatoY = formatoX if (formatoX ** 2 - len(imagens)) <= formatoX else formatoX - 1

    # Mostra imagens
    for i in range(len(imagens)):
        plt.subplot(formatoY, formatoX, i + 1)
        plt.imshow(imagens[i], 'gray' if len(imagens[i].shape) == 2 else None)
        plt.title(titles[i], fontsize=8)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle(f"Resultado - {nome_img}", fontsize=10)
    plt.tight_layout()
    plt.show()