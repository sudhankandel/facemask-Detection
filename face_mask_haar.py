import cv2
# from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import model_from_json
import numpy as np


import pygame

pygame.init()

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('model_weight.h5')


cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    fcs=[]
    cfd=[]
    preds=[]
    locs=[]

    # mt=MTCNN()
    # faces=mt.detect_faces(frame)
    # for face in faces:
    #     x,y,w,h = face['box']
    #     confidence = face['confidence']
    #     x2=x+w
    #     y2=y+h
    #     fc=frame[y:y2, x:x2]
    #     fc= cv2.resize(fc, (100,100))
    #     fcs.append(fc)
    #     locs.append((x, y, x2, y2))
    #     cfd.append(confidence)



    
    hcfaces= facec.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in hcfaces:
        x2=x+w
        y2=y+h
        fc=frame[y:y2,x:x2]
        fc= cv2.resize(fc, (100,100))
        fcs.append(fc)
        locs.append((x, y, x2, y2))


        conf=0
        if len(fcs)>0:
            fcs=np.array(fcs,dtype='float')
            preds=loaded_model.predict(fcs)
        
        for (bbox,pred) in zip(locs,preds):
            (x,y,x2,y2)=bbox
            label='mask' if pred[0]>0.5 else 'No Mask'
            if label=='mask':
                conf=pred[0]*100
            else:
                conf=(100-pred[0]*100) 
                pygame.mixer.music.load("sound.mp3")
                pygame.mixer.music.play()

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # green if mask else red
            label = "{}: {:.2f}%".format(label, conf) # % of confidence


            cv2.rectangle(frame, (x, y), (x2, y2), color, 3)  # Putting rectangle of bbox in frames
            cv2.rectangle(frame, (x,y-40), (x2,y), color, -1)

            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Putting text in live frames
    cv2.imshow('frame', frame)   
    

    if (cv2.waitKey(20) == ord('q')) or (cv2.waitKey(20) == 27):
            
            break

cap.release()
cv2.destroyAllWindows()