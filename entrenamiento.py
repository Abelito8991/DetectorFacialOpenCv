import cv2
import os
import numpy as np

ruta = os.getcwd()+"\\Data"
personaList = os.listdir(ruta)


labels = []
faceData = []
label = 0

for nameDir in personaList:
    rutaPersona = ruta + "\\" + nameDir

    for filename in os.listdir(rutaPersona):
        labels.append(label)
        faceData.append(cv2.imread(rutaPersona+"\\"+filename,0))

    label = label + 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faceData, np.array(labels))
face_recognizer.write("modeloLBPHFACE.xml")