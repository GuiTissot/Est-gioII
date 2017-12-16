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
imagem_svrna = cv2.imread("C:\Users\GuilhermeHoffmannTis\Desktop/svrna_teste3.jpg", )



##REDIMENCIONAMENTO
redimencionado = cv2.resize(imagem, (300, 200), interpolation=cv2.INTER_AREA)
redimencionado_circulo = cv2.resize(imagem_circulo, (300, 200), interpolation=cv2.INTER_AREA)
redimencinado_svrna = cv2.resize(imagem_svrna,(300, 200), interpolation=cv2.INTER_AREA)

print "Altura (height): %d pixels" % (redimencionado.shape[0])
print "Largura (width): %d pixels" % (redimencionado.shape[1])

##FILTRO



limiar = 127
maximo = 255

## SVRNA

kernels = np.ones((1,1),np.uint8)
grays = cv2.cvtColor(imagem_svrna,cv2.COLOR_BGR2GRAY)
mediana = cv2.medianBlur(grays, 1)
ret, threshsvrna = cv2.threshold(mediana, limiar, maximo, cv2.THRESH_BINARY)


openingsvrna = cv2.morphologyEx(threshsvrna,cv2.MORPH_OPEN,kernels, iterations = 2)
#closingsvrna = cv2.morphologyEx(threshsvrna,cv2.MORPH_CLOSE,kernels, iterations = 2)




# PIXEL

imagem_data = np.asarray(openingsvrna)

# Percorrer a imagem
for i in range(len(imagem_data)):
    for j in range(len(imagem_data[0])):
        imagem_data[i][j]

fase_escura = np.mean(imagem_data / 255)  # CALCULO FASE ESCURA
print "Fase Escura : " + str(fase_escura * 100) + '%'
aux = fase_escura* 100
fase_clara = 100 - aux
print "Fase clara : " + str(fase_clara) + '%'  # CALCULO FASE CLARA




##TESTES SEGUNDO PLOT
ret, thresh1 = cv2.threshold(redimencionado, limiar, maximo, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(redimencionado, limiar, maximo, cv2.THRESH_TRUNC)
#thresh3 = cv2.adaptiveThreshold(imagem_svrna, maximo, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                        #cv2.THRESH_BINARY, 11, 2)
#blur = cv2.bilateralFilter(thresh3, 9, 75, 75)

ret, thresh4 = cv2.threshold(redimencionado, limiar, maximo, cv2.THRESH_TOZERO)

#ret, thresh5 = cv2.threshold(thresh3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ret,thresh5 = cv2.threshold(redimencionado,limiar,maximo,cv2.THRESH_TOZERO_INV)
# ret,thresh6 = cv2.threshold(redimencionado,127,255,cv2.THRESH_BINARY_INV)



##TESTES WATERSHED - FUNCINANDO

img = cv2.imread('C:\Users\GuilhermeHoffmannTis\Desktop\Vanadis4_1000x vilela.jpg')
print "Altura (height): %d pixels" % (img.shape[0])
print "Largura (width): %d pixels" % (img.shape[1])
print "Canais (channels): %d"      % (img.shape[2])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#Estimando a localizacao dos graos

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Removendo o ruido

kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

# Dilatacao

background = cv2.dilate(closing,kernel,iterations=3)

# Area de Fundo
dist_transform = cv2.distanceTransform(background,cv2.DIST_L2,3)

# Threshold
ret, foreground = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Procurando os valores dos locais desconhecidos
foreground = np.uint8(foreground)
unknown = cv2.subtract(background,foreground)

# Aplicando o Markers
ret, markers = cv2.connectedComponents(foreground)


# Aplicando 1 a todo plano de fundo
markers = markers+1

# Agora marcando a regiao  (bordas) com 0
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]



plt.subplot(421),plt.imshow(gray)
plt.title('IMAGEM DE ENTRADA'), plt.xticks([]), plt.yticks([])

plt.subplot(422),plt.imshow(thresh, 'gray')
plt.title("TRESHHOLD BINARIO OTSU'S"), plt.xticks([]), plt.yticks([])

plt.subplot(423),plt.imshow(closing, 'gray')
plt.title("MORFOLOGIA:CLOSING:2x2"), plt.xticks([]), plt.yticks([])

plt.subplot(424),plt.imshow(background, 'gray')
plt.title("DILATACAO"), plt.xticks([]), plt.yticks([])

plt.subplot(425),plt.imshow(dist_transform, 'gray')
plt.title("TRANFORMACAO DE DISTANCIA"), plt.xticks([]), plt.yticks([])

plt.subplot(426),plt.imshow(foreground, 'gray')
plt.title("THRESHOLDING"), plt.xticks([]), plt.yticks([])

plt.subplot(427),plt.imshow(unknown, 'gray')
plt.title("DESCONHECIDO"), plt.xticks([]), plt.yticks([])

plt.subplot(428),plt.imshow(img, 'gray')
plt.title("RESULTADO DE WATERSHEED"), plt.xticks([]), plt.yticks([])



plt.tight_layout()
plt.show()

##CIRCULO

img_rez=cv2.resize(img, (1000, 700), interpolation=cv2.INTER_AREA)
diametro = 300  # Diametro tem que ser 79.8
height, width = img.shape[ :2]
center = (height/2 , width/2)
print center
cv2.circle(img_rez, (center), diametro, preto, 1)

##TITULOS
titles = ['Imagem Original', 'CLOSINGSVRNA', 'MEDIANA', 'SRVNAGRAY', 'TESTE', "TRUNC"]
images = [threshsvrna,      openingsvrna,    mediana,      grays , closing, thresh2]

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
