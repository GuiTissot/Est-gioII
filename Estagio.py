import cv2
from matplotlib import pyplot as plt
from PIL import Image
from sys import argv
import numpy as np

preto = (0, 0, 0)

# IMAGENS

imagem = cv2.imread("C:\Users\GuilhermeHoffmannTis\Desktop\Vanadis4_1000x vilela.jpg", 0)  # Receber a Imagem
imagem_circulo = cv2.imread("C:\Users\GuilhermeHoffmannTis\Desktop\Vanadis4_1000x vilela.jpg", 0)
imagem_cor = cv2.imread("C:\Users\GuilhermeHoffmannTis\Desktop\Vanadis4_1000x vilela.jpg", )

##REDIMENCIONAMENTO
redimencionado = cv2.resize(imagem, (300, 200), interpolation=cv2.INTER_AREA)
redimencionado_circulo = cv2.resize(imagem_circulo, (300, 200), interpolation=cv2.INTER_AREA)

print "Altura (height): %d pixels" % (redimencionado.shape[0])
print "Largura (width): %d pixels" % (redimencionado.shape[1])

##FILTRO


# limiar = 127
limiar = 127
maximo = 255

ret, thresh1 = cv2.threshold(redimencionado, limiar, maximo, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(redimencionado, limiar, maximo, cv2.THRESH_TRUNC)
thresh3 = cv2.adaptiveThreshold(redimencionado, maximo, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 11, 2)
blur = cv2.bilateralFilter(thresh3, 9, 75, 75)

ret, thresh4 = cv2.threshold(redimencionado, limiar, maximo, cv2.THRESH_TOZERO)
mediana = cv2.medianBlur(redimencionado, 5)
ret, thresh5 = cv2.threshold(thresh3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ret,thresh5 = cv2.threshold(redimencionado,limiar,maximo,cv2.THRESH_TOZERO_INV)
# ret,thresh6 = cv2.threshold(redimencionado,127,255,cv2.THRESH_BINARY_INV)

##CIRCULO
diametro = 80  # Diametro tem que ser 79.8

cv2.circle(redimencionado_circulo, (150, 100), diametro, preto, 1)

# PIXEL

imagem_data = np.asarray(thresh1)

# Percorrer a imagem
for i in range(len(imagem_data)):
    for j in range(len(imagem_data[0])):
        imagem_data[i][j]

fase_clara = np.mean(imagem_data / 255)  # CALCULO FASE CLARA
print str(fase_clara * 100) + '%'
aux = fase_clara * 100
fase_escura = 100 - aux
print str(fase_escura) + '%'  # CALCULO FASE ESCURA



##TESTES Watershed - MDS DEU CERTO FINALMENTE



img = cv2.imread('C:\Users\GuilhermeHoffmannTis\Desktop\Vanadis4_1000x vilela.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Removendo os ruidos
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Area de Fundo
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Procurando a area de primeiro plano da imagem
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Procurando os valores dos locais desconhecidos
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Aplicando o Markers
ret, markers = cv2.connectedComponents(sure_fg)

# Aplicando 1 a todo plano de fundo
markers = markers + 1

# Agora marcando a regiao desconhecida com 0
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]





##TESTES WATERSHED - TALVEZ FUNCIONE



gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Removendo o ruido
kernel = np.ones((2,2),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

# Procurando a area de primeiro plano da imagem
sure_bg = cv2.dilate(closing,kernel,iterations=3)

# Area de Fundo
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

# Threshold
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Procurando os valores dos locais desconhecidos
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Aplicando o Markers
ret, markers = cv2.connectedComponents(sure_fg)


# Aplicando 1 a todo plano de fundo
markers = markers+1

# Agora marcando a regiao desconhecida com 0
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.subplot(421),plt.imshow(gray)
plt.title('IMAGEM DE ENTRADA'), plt.xticks([]), plt.yticks([])
plt.subplot(422),plt.imshow(thresh, 'gray')
plt.title("TRESHHOLD BINARIO OTSU'S"), plt.xticks([]), plt.yticks([])

plt.subplot(423),plt.imshow(closing, 'gray')
plt.title("MORFOLOGIA:CLOSING:2x2"), plt.xticks([]), plt.yticks([])
plt.subplot(424),plt.imshow(sure_bg, 'gray')
plt.title("DILATACAO"), plt.xticks([]), plt.yticks([])

plt.subplot(425),plt.imshow(dist_transform, 'gray')
plt.title("TRANFORMACAO DE DISTANCIA"), plt.xticks([]), plt.yticks([])
plt.subplot(426),plt.imshow(sure_fg, 'gray')
plt.title("THRESHOLDING"), plt.xticks([]), plt.yticks([])

plt.subplot(427),plt.imshow(unknown, 'gray')
plt.title("DESCONHECIDO"), plt.xticks([]), plt.yticks([])

plt.subplot(428),plt.imshow(img, 'gray')
plt.title("RESULTADO DE WATERSHEED"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()


##TITULOS
titles = ['Imagem Original', 'BINARIO', 'MEDIANA', 'GAUSSIANO', 'TESTE', "TRUNC"]
images = [imagem, thresh1, mediana, thresh3, closing, thresh2]

##INTERSECCOES



# filename = cv2.imread("C:\Users\GuilhermeHoffmannTis\Desktop\Vanadis4_1000x vilela.jpg",0)

# I = cv2.imread(filename, 0)
# I = cv2.medianBlur(I, 3)
# bw = cv2.adaptiveThreshold(I, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
# bw = cv2.dilate(cv2.erode(bw, kernel), kernel)
# print np.round_(np.sum(bw == 0) / 3015.0)


##PLOT

for i in xrange(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
