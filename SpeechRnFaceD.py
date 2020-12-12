# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:04:25 2020

@author: Tejas

## Requirements:
- !pip install SpeechRecognition
- !pip install cv2
- !pip install pyaudio
- !conda install -c conda-forge dlib   

- Download a file from here- https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat

"""

# import speech_recognition as sr
import numpy as np
import cv2
import dlib
import math
import time


def StartVideo():
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("Window")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    count = 0
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ## Facial detection
        faces = detector(gray)
        #print(len(faces)) # =1
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1) #1 pixel thickness
            
            landmarks = predictor(gray, face)
            
            '''
            ### Find the ears upper and lower region
            #for n in range(0, 68):
            for n in [0, 2, 14, 16]:      ### Ears = [0,2,14,16]
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x,y), 3, (255, 0, 0), -1) #radius=3, fill the circle = -1
            '''   
        
              
        ### Drawing a line from point 28 to 8.
        x28 = landmarks.part(28).x
        y28 = landmarks.part(28).y
        x8 = landmarks.part(8).x
        y8 = landmarks.part(8).y
        
        
        ### Calculate angle between the 2 points
        angle = math.atan2((y8-y28),(x8-x28))*180/3.141592653;
        angle = angle + math.ceil(-angle / 360 ) * 360
        
        allx = []
        ally = []
        if angle > 85 and angle < 95:
            
            x = landmarks.part(66).x
            y = landmarks.part(66).y
            
            allx.append(x)
            ally.append(y)
            
            cv2.putText(frame, "Perfect, stay still!, mouth location = ({},{})".format(x, y),
                    (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            

            cv2.circle(frame, (x,y), 4, (255, 0, 0), -1) #radius=3, fill the circle = -1 
            #return x, y
        
        else:
            
            ### Facial Center points forehead, lower lip center and chin. Points 28, 57 and 8.
            for n in [28, 57, 8]:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x,y), 4, (255, 0, 0), -1) #radius=3, fill the circle = -1
            
            
            cv2.line(frame, (x28, y28), (x8,y8), (0,0,0), 2) 


            ### Tell the user to adjust the face orientation
            #cv2.putText(img,'Hello World!', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
            cv2.putText(frame,'Adjust face to 90 Deg, current angle = {}'.format(angle),
                        (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                
        
        cv2.imshow("Window", frame)

        #This breaks on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return math.floor(np.mean(allx)), math.floor(np.mean(ally))


x, y = StartVideo()

print(x, y)



# r = sr.Recognizer()
# with sr.Microphone() as source:
#     print('Start Speaking....')
#     audio = r.listen(source, phrase_time_limit=3)
#     print("Recording Ended!")
    
# try:
#     text = r.recognize_google(audio)
#     print(text)
# except:
#     print("Could Not Understand the audio!, Retry again!")
    
    
# if text == 'hi hello':
#     x, y = StartVideo()
















