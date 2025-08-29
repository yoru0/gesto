import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands   = mpHands.Hands()
mpDraw  = mp.solutions.drawing_utils

prevTime = 0

while True:
    ok, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10, 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
