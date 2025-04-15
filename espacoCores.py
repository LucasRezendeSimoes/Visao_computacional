# pip install opencv-python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Lista de imagens
nomes_imgs = ['GIRAFA.jpg', 'Satelite.jpg', 'Aviao.jpg']

# Loop pelas imagens
for nome_img in nomes_imgs:
    #valores
    if nome_img == 'GIRAFA.jpg':
        brilho = 1.0
        contraste = 1.0
    elif nome_img == 'Satelite.jpg':
        brilho = 1.2
        contraste = 1.1
    elif nome_img == 'Aviao.jpg':
        brilho = 1.1
        contraste = 1.0

    # Carrega a imagem
    img = cv2.imread(f'./Images/{nome_img}')

    # Converte BGR para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Aplica brilho e contraste
    img_rgb = cv2.convertScaleAbs(img_rgb, alpha=contraste, beta=brilho * 10)

    # Converte para cinza, HSV e HLS
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Junta as imagens
    imagens = [img_rgb, img_gray, img_hsv, img_hls]
    titles = [f"{nome_img} - RGB", "Escala de cinza", "HSV", "HLS"]

    # Plot
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(imagens[i], 'gray' if len(imagens[i].shape) == 2 else None)
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()