import cv2
import time
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class HandData:
    lm_list: List[Tuple[int, int, int]]
    label: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]


class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self._last_handedness = []

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        self._last_handedness = []
        if self.results and self.results.multi_handedness:
            for h in self.results.multi_handedness:
                self._last_handedness.append(h.classification[0].label)

        if self.results and self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def _lm_to_pixels(self, img, hand_no=0):
        lm_list = []
        my_hand = self.results.multi_hand_landmarks[hand_no]
        h, w, c = img.shape
        for id, lm in enumerate(my_hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((id, cx, cy))
        return lm_list

    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results and self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                lm_list = self._lm_to_pixels(img, hand_no)
                if draw:
                    for _, cx, cy in lm_list:
                        cv2.circle(img, (cx, cy), 1, (0, 0, 0), cv2.FILLED)
        return lm_list
    
    def get_hands(self, img, draw_dots=False) -> List[HandData]:
        hands_out: List[HandData] = []
        if not (self.results and self.results.multi_hand_landmarks):
            return hands_out

        for idx, _ in enumerate(self.results.multi_hand_landmarks):
            lm_list = self._lm_to_pixels(img, idx)
            xs = [p[1] for p in lm_list]; ys = [p[2] for p in lm_list]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            center = (int(sum(xs)/len(xs)), int(sum(ys)/len(ys)))
            label = self._last_handedness[idx] if idx < len(self._last_handedness) else "Right"

            if draw_dots:
                for _, cx, cy in lm_list:
                    cv2.circle(img, (cx, cy), 2, (0, 0, 0), cv2.FILLED)
                x, y, w, h = bbox
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 1)
                cv2.putText(img, label, (x, y-6), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

            hands_out.append(HandData(lm_list=lm_list, label=label, bbox=bbox, center=center))
        return hands_out


def main():
    prev_time, curr_time = 0, 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)
        if len(lm_list) != 0:
            print(lm_list[:5])

        hands = detector.get_hands(img)
        if hands:
            cx, cy = hands[0].center
            cv2.circle(img, (cx, cy), 4, (0, 0, 0), -1)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, str(int(fps)), (10, 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()