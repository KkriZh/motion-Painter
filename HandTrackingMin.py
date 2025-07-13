import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands() #only use RGB images
mpDraw=mp.solutions.drawing_utils
ptime=0           #previous time
ctime=0           #current time

while True:
    success, img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convert to RGB
    results=hands.process(imgRGB)
    #print(results.multi_hand_landmarks)#check if something is detected or not
    
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                print(id,lm)
                h,w,c=img.shape #hieght, width, channel of image
                cx,cy=int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                #if id==0:
                cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)#circle the landmark

            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)#draw hand landmarks, draw connections

    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3) #(img,fps,position(x,y), FONT, scale, color(a,b,c),thickness )
    cv2.imshow("Image",img)
    cv2.waitKey(1)

     