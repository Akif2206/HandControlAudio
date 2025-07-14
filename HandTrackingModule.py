import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []

        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                lmList = []
                xList = []
                yList = []

                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    xList.append(cx)
                    yList.append(cy)

                bbox = [min(xList), min(yList), max(xList), max(yList)]
                myHand["lmList"] = lmList
                myHand["bbox"] = bbox
                myHand["type"] = handType.classification[0].label  # 'Left' or 'Right'

                allHands.append(myHand)

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img, allHands

    def fingersUp(self, lmList, handType="Right"):
        fingers = []
        # Thumb
        if handType == "Right":
            fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)
        else:
            fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)

        # 4 fingers
        for tip in [8, 12, 16, 20]:
            fingers.append(1 if lmList[tip][2] < lmList[tip - 2][2] else 0)
        return fingers
