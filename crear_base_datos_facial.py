import cv2
import os

#nombre de la persona que queremos crear la base de datos
persona = "Abel"
ruta=os.getcwd()+"\\Data"
rutaPersona=ruta+"\\"+persona

#creamos la carpeta de esta persona si no existe
if not os.path.exists(rutaPersona):
    os.makedirs(rutaPersona)

#cargamos el clasificador de cara
faceClasif = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
cont=0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = cv2.resize(frame, None, fx=1, fy=1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClasif.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro=auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(rutaPersona+"\\rostro_{}.jpg".format(cont), rostro)
        cont=cont+1
    cv2.imshow("frame", frame)


    k=cv2.waitKey(1)
    if k==27 or cont >=300:
        break

cap.release()
cv2.destroyAllWindows()