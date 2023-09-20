import cv2
import time
import numpy as np
import HandTrackingModule as htm

###################################
wCam, hCam = 640, 320
###################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(detection_conf=0.9)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), radius=8, color=(0, 255, 0), thickness=cv2.FILLED)
        cv2.circle(img, (x2, y2), radius=8, color=(0, 255, 0), thickness=cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)
        cv2.circle(img, (cx, cy), radius=10, color=(0, 0, 255), thickness=cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
