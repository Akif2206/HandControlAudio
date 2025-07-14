import cv2
import time
import math
import numpy as np
import HandTrackingModule as htm
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Webcam and hand detector
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon=0.8)

# Volume control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
minVol, maxVol = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img, hands = detector.findHands(img)

    leftHand, rightHand = None, None
    for hand in hands:
        if hand["type"] == "Left":
            leftHand = hand
        elif hand["type"] == "Right":
            rightHand = hand

    # Volume (index-to-index distance)
    if leftHand and rightHand:
        x1, y1 = leftHand["lmList"][8][1], leftHand["lmList"][8][2]
        x2, y2 = rightHand["lmList"][8][1], rightHand["lmList"][8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)
        volPercent = int(np.interp(length, [50, 300], [0, 100]))

        # Volume animation between hands
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        for i in range(0, 11):
            alpha = i / 10
            ix = int((1 - alpha) * x1 + alpha * x2)
            iy = int((1 - alpha) * y1 + alpha * y2)
            cv2.circle(img, (ix, iy), 5, (0, int(255 * alpha), 255 - int(255 * alpha)), -1)

        cv2.putText(img, f'Volume: {volPercent}%', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Bass (distance between thumb & index on left hand)
    if leftHand:
        xL1, yL1 = leftHand["lmList"][4][1], leftHand["lmList"][4][2]  # Thumb
        xL2, yL2 = leftHand["lmList"][8][1], leftHand["lmList"][8][2]  # Index
        bassLength = math.hypot(xL2 - xL1, yL2 - yL1)
        bassPercent = int(np.interp(bassLength, [20, 150], [0, 100]))
        cv2.line(img, (xL1, yL1), (xL2, yL2), (0, 255, 0), 2)
        cv2.putText(img, f'Bass: {bassPercent}%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Frequency (distance between thumb & index on right hand)
    if rightHand:
        xR1, yR1 = rightHand["lmList"][4][1], rightHand["lmList"][4][2]  # Thumb
        xR2, yR2 = rightHand["lmList"][8][1], rightHand["lmList"][8][2]  # Index
        freqLength = math.hypot(xR2 - xR1, yR2 - yR1)
        freqPercent = int(np.interp(freqLength, [20, 150], [0, 100]))
        cv2.line(img, (xR1, yR1), (xR2, yR2), (0, 255, 255), 2)
        cv2.putText(img, f'Freq: {freqPercent}%', (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("Hand DJ Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
