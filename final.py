import numpy as np
import cv2
from collections import deque
import serial

camara = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

contador = 0
(dx, dy) = (0,0)
direccion = ""

enviar = serial.Serial('COM3', 115200, None)

while True:

    (_, cuadro) = camara.read()
    gris = cv2.cvtColor(cuadro, cv2.COLOR_RGB2GRAY)
    borr = cv2.GaussianBlur(gris, (11,11), 0)
    copia = cuadro.copy()
    
    deteccion = cascade.detectMultiScale(borr, 1.3, 20)

    for (x,y,w,h) in deteccion:
        cv2.rectangle(copia, (x,y), ((x+w), (y+h)), (0,255,0), 1)

        corte = borr[y:y+h, x:x+w]
        vent = cv2.resize(corte, (340, 340))

        masc = cv2.inRange(vent, 0, 26)
        ero = cv2.erode(masc, None, iterations = 2)
        dil = cv2.dilate(ero, None, iterations =2)
        canny = cv2.Canny(dil, 80, 90)
        
        copia1 = cuadro[y:y+h, x:x+w]
        salida = cv2.resize(copia1, (340, 340))

        (_, cont, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cont) >0:
            c = max(cont, key=cv2.contourArea)
            ((x,y),r) = cv2.minEnclosingCircle(c)
            centro = cv2.moments(c)
            ctr = (int(centro['m10']/centro['m00']), int(centro['m01']/centro['m00']))
            cx = int(centro['m10']/centro['m00'])
            cy = int(centro['m01']/centro['m00'])
            if r >26:
                cv2.circle(salida, ctr, 10, (255,255,255), -1)

                (dirX, dirY) = ("", "")

                if cx > 200:
                    dirX = "Izquierda"
                    enviar.write('a')
                if cx < 150:
                    dirX = "Derecha"
                    enviar.write('b')

                if cy < 170:
                    dirY = "Arriba"
                    enviar.write('c')

                if dirX != "" and dirY != "":
                    direccion = "{}-{}".format(dirY,dirX)
                else:
                    direccion = dirX if dirX != "" else dirY

    cv2.putText(salida, direccion, (50,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
               (0,0,255), 3)  
        
    cv2.imshow('prubea 3', salida)
    key = cv2.waitKey(1) & 0xFF
    contador += 1
    
    if key == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()
#enviar.close()
