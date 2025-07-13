import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode #we are going to create objrct and it create its own variable
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        self.mpHands=mp.solutions.hands
        self.hands = self.mpHands.Hands(
                        static_image_mode=self.mode,
                        max_num_hands=self.maxHands,
                        model_complexity=0,  # optional, default is 1
                        min_detection_confidence=self.detectionCon,
                        min_tracking_confidence=self.trackCon
                    )

        self.mpDraw=mp.solutions.drawing_utils
        self.tipIds=[4,8,12,16,20]


    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convert to RGB
        self.results=self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)#check if something is detected or not
        
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)#draw hand landmarks, draw connections
        return img



    def findPosition(self,img,handNo=0, draw=True):
        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo] #get the hand
            for id, lm in enumerate(myHand.landmark): #get all the landmarks within the hand
                    #print(id,lm)
                    h,w,c=img.shape #hieght, width, channel of image
                    cx,cy=int(lm.x*w), int(lm.y*h)
                    #print(id,cx,cy)
                    self.lmList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),7,(0, 255, 80),cv2.FILLED)#circle the landmark
        return self.lmList

    def fingersUp(self):
        fingers=[]
        #thumb
        if self.lmList[self.tipIds[0]][1]<self.lmList[self.tipIds[0]-1][1]: #check if tip of thumb is right or left and tell open or close, it is for left hand orientation but if img is fliped then right hand orientation
            fingers.append(1)
        else:
            fingers.append(0)
        
        #4 fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2]<self.lmList[self.tipIds[id]-2][2]: #check if above lm 2 step below it or not
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
            


def main():
    ptime=0           #previous time
    ctime=0           #current time
    cap=cv2.VideoCapture(0)
    detector=handDetector()
    while True:
        success, img=cap.read()
        img=detector.findHands(img,draw=True) #gets the detected image
        lmList=detector.findPosition(img,draw=True)# draw=False if want to remove drawing in both cases
        #if len(lmList) !=0:
            #print(lmList[4])
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0, 255, 200),3) #(img,fps,position(x,y), FONT, scale, color(a,b,c),thickness )
        cv2.imshow("Image",img)
        cv2.waitKey(1)




if __name__=="__main__":
    main()