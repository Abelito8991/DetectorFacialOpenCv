import cv2
import os

ruta = os.getcwd()+"\\Data"
rutasImagen = os.listdir(ruta)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("modeloLBPHFACE.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClasif = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if ret==False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClasif.detectMultiScale(gray, 1.3, 5)
    if len(faces) != 0:
        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            if result[1]<70:
                cv2.putText(frame, "{}".format(rutasImagen[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.putText(frame, "Desconocido",(x,y-25),2,0.8,(255,0,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    else:
        cv2.destroyAllWindows()
    
cap.release()
cv2.destroyAllWindows()