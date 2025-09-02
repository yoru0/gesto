import cv2
import time
import numpy as np
import hand_tracking_module as htm

cam_width, cam_height = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

prev_time = 0

detector = htm.HandDetector(detection_con=0.7)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        print(lm_list[8], lm_list[12])  
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (10, 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()