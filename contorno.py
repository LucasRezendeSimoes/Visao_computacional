import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Lista com nomes das imagens
imagens_input = ['GIRAFA.jpg', 'Satelite.jpg', 'Aviao.jpg']

# Loop para processar cada imagem
for nome_img in imagens_input:
    # Lê a imagem e converte para RGB
    img = cv2.imread(f'./Images/{nome_img}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Converte para escala de cinza
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Define parâmetros por imagem
    if nome_img == 'GIRAFA.jpg':
        fator_thresh = 1.6
        tamanhoKernel = 5
        thresh1 = img_gray.max() * 0.4
        thresh2 = img_gray.max() * 0.6
    elif nome_img == 'Satelite.jpg':
        fator_thresh = 1.4
        tamanhoKernel = 7
        thresh1 = img_gray.max() * 0.5
        thresh2 = img_gray.max() * 0.6
    elif nome_img == 'Aviao.jpg':
        fator_thresh = 1.65
        tamanhoKernel = 6
        thresh1 = img_gray.max() * 0.5
        thresh2 = img_gray.max() * 0.5

    # Aplica threshold binário invertido
    a = img_gray.max()
    _, thresh = cv2.threshold(img_gray, a/2 * fator_thresh, a, cv2.THRESH_BINARY_INV)

    # Kernel
    kernel = np.ones((tamanhoKernel, tamanhoKernel), np.uint8)

    # Abertura morfológica
    thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Blur
    img_blur = cv2.blur(img_gray, ksize=(tamanhoKernel, tamanhoKernel))

    # Canny (sem e com blur)
    edges_gray = cv2.Canny(image=img_gray, threshold1=thresh1, threshold2=thresh2)
    edges_blur = cv2.Canny(image=img_blur, threshold1=thresh1, threshold2=thresh2)

    # Contornos
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_copy = img.copy()
    final = cv2.drawContours(img_copy, contours, contourIdx=-1, color=(255, 0, 0), thickness=2)

    # Lista de imagens e títulos
    imagens = [img, img_blur, img_gray, edges_gray, edges_blur, thresh, thresh_open, final]
    titles = [
        "Imagem original", 
        "Imagem com blur", 
        "Escala de cinza", 
        "Canny sem blur", 
        "Canny com blur", 
        "Threshold binário", 
        "Abertura morfológica", 
        "Contornos detectados"
    ]

    # Grid automático
    formatoX = math.ceil(len(imagens) ** 0.5)
    formatoY = formatoX if (formatoX**2 - len(imagens)) <= formatoX else formatoX - 1

    # Plota as imagens
    for i in range(len(imagens)):
        plt.subplot(formatoY, formatoX, i + 1)
        plt.imshow(imagens[i], 'gray' if len(imagens[i].shape) == 2 else None)
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])

    # Layout final
    plt.suptitle(f"Resultado para {nome_img}", fontsize=10)
    plt.tight_layout()
    plt.show()