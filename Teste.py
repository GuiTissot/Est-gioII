import cv2
from matplotlib import pyplot as plt

limiar = 127
maximo = 255
preto = (0, 0, 0)
diametro = 80 #Diametro tem que ser 79.8

imagem = cv2.imread("C:\Users\GuilhermeHoffmannTis\Desktop\Imagem_Metalografica.jpeg",0) #Receber a Imagem
imagem_circulo = cv2.imread("C:\Users\GuilhermeHoffmannTis\Desktop\Imagem_Metalografica.jpeg",0)

print "Altura (height): %d pixels" % (imagem.shape[0])
print "Largura (width): %d pixels" % (imagem.shape[1])



ret,thresh1 = cv2.threshold(imagem, limiar, maximo, cv2.THRESH_BINARY)
teste = cv2.circle(imagem_circulo, (150, 104), diametro, preto, 1)
titles = ['Imagem Original','BINARIO','CIRCULO']
images = [imagem, thresh1,teste]

for i in xrange(3):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


cv2.waitKey(0)