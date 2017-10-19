import cv2

imagem = cv2.imread("C:\Users\GuilhermeHoffmannTis\Desktop\Imagem_Metalografica.jpeg") #Receber a Imagem


print "Altura (height): %d pixels" % (imagem.shape[0])
print "Largura (width): %d pixels" % (imagem.shape[1])
print "Canais (channels): %d" % (imagem.shape[2]) #Armarzena a quantidade de cor, no caso 3 sendo RGB


(b, g, r) = imagem[0, 0]
print "Cor do pixel em (0, 0) - Vermelho: %d, Verde: %d, Azul: %d" % (r, g, b) #Cor Preta
(b, g, r) = imagem[200, 210]
print "Cor do pixel em (250, 305) - Vermelho: %d, Verde: %d, Azul: %d" % (r, g, b) #Cor Branca
(b, g, r) = imagem[30, 220]
print "Cor do pixel em (250, 30) - Vermelho: %d, Verde: %d, Azul: %d" % (r, g, b) #Cor Cinza


preto = (0, 0, 0)
diametro = 80 #Diametro tem que ser 79.8
cv2.circle(imagem, (150, 104), diametro, preto, 1) #Desenhar o Ciculo centralizado Primeiro e Segundo Parametro - Local do Circulo, Terceiro Parametro - Diametro do circulo
cv2.imshow("Teste", imagem) #Mostrar a imagem


cv2.waitKey(0)