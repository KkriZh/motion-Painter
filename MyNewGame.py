import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

ptime = 0  # previous time
ctime = 0  # current time
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)  # detect hands
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        print(lmList[4])  # print thumb tip position

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)

    # 👇 Clean exit: press 'q' or ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Exiting Hand Tracking...")
        break

# 🔚 Release webcam and close window
cap.release()
cv2.destroyAllWindows()
