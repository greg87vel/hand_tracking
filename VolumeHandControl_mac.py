import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import subprocess


def get_volume():
    cmd = "osascript -e 'output volume of (get volume settings)'"
    return int(subprocess.check_output(cmd, shell=True))


def set_volume(volume):
    cmd = f"osascript -e 'set volume output volume {volume}'"
    subprocess.call(cmd, shell=True)


###################################
wCam, hCam = 640, 480
###################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(detection_conf=0.9)

minVol = 0
maxVol = 100  # Il volume in macOS va da 0 a 100
vol = get_volume()
volBar = 400
volPer = 0
relLength = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        xp, yp = lmList[4][1], lmList[4][2]
        xi, yi = lmList[8][1], lmList[8][2]
        cx, cy = (xp + xi) // 2, (yp + yi) // 2
        xpolso, ypolso = lmList[0][1], lmList[0][2]
        xbm, ybm = lmList[9][1], lmList[9][2]

        cv2.circle(img, (xp, yp), radius=8, color=(0, 255, 0), thickness=cv2.FILLED)
        cv2.circle(img, (xi, yi), radius=8, color=(0, 255, 0), thickness=cv2.FILLED)
        cv2.line(img, (xp, yp), (xi, yi), (255, 255, 255), thickness=3)
        cv2.circle(img, (cx, cy), radius=10, color=(0, 0, 255), thickness=cv2.FILLED)

        pi_Length = math.hypot((xi - xp), (yi - yp))
        polsobm_Length = math.hypot((xbm - xpolso), (ybm - ypolso))
        relLength = pi_Length / polsobm_Length

        min_relLength = 0.2
        max_relLength = 1.4

        vol = np.interp(relLength, [min_relLength, max_relLength], [minVol, maxVol])
        volBar = np.interp(relLength, [min_relLength, max_relLength], [400, 150])
        volPer = np.interp(relLength, [min_relLength, max_relLength], [0, 100])

        set_volume(int(volPer))

        if relLength < min_relLength:
            cv2.circle(img, (cx, cy), radius=10, color=(0, 255, 255), thickness=cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
