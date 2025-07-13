import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath="Header"
myList=os.listdir(folderPath)
print(myList)
overlayList=[]
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header=overlayList[0]
cap=cv2.VideoCapture(0)#If the webcam doesn't work (`VideoCapture(0)` returns black screen), try:cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)
drawColor=(255,0,0)
brushThickness=10
eraserThickness=100

detector=htm.handDetector(detectionCon=0.85)
xp,yp=0,0
# Create a blank (black) image canvas of size 720x1280 with 3 color channels (RGB), dtype should be np.uint8
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:

    #1.Import image
    success, img=cap.read()
    img=cv2.flip(img,1) # flip image 1-sidewise 0-upsidedown

    #2.Find Hand Landmarks
    img=detector.findHands(img)
    lmList=detector.findPosition(img, draw=False)

    if len(lmList) !=0:
        
        #print(lmList)
        #Tip of index and middle fingers
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]

        #3. check which finger is up
        fingers=detector.fingersUp()
        #print(fingers)

        #4. if selection mode= two fingers are up
        if fingers[1] == 1 and fingers[2] == 1:
            xp,yp=0,0 # reset hand whenever hand is detected
            print("selection mode")
            #checking the click
            if y1<125:
                if 150<x1<350:
                    header=overlayList[0]
                    drawColor=(255,0,0)#blue
                elif 450<x1<650:
                    header=overlayList[1]
                    drawColor=(255,0,255)#pink
                elif 700<x1<850:
                    header=overlayList[2]
                    drawColor=(0,255,0)#green
                elif 950<x1<1100:
                    header=overlayList[3]
                    drawColor=(0,0,0)#black (BGR)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)

        #5. if drawing mode= index finger is up
        elif fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("drawing mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1

            if drawColor==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            
            xp,yp=x1,y1

    # Convert the canvas (imgCanvas) from BGR to Grayscale format
    # This simplifies it for thresholding
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

    # Apply inverse binary thresholding:
    # Pixels with intensity > 50 become 0 (black), others become 255 (white)
    # This creates a mask: drawings turn black, background turns white
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)

    # Convert the single-channel imgInv (grayscale) back to BGR so it matches the original image
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    # Bitwise AND: removes the drawing area from the original webcam image using the inverse mask
    img = cv2.bitwise_and(img, imgInv)

    # Bitwise OR: combines the original image (with drawing area cleared) and the canvas (with drawing)
    img = cv2.bitwise_or(img, imgCanvas)


    #setting header image
    img[0:125,0:1280]=header
    #img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.imshow("canvas", imgCanvas)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' or 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
