import cv2
import numpy as np
import json, time, collections
import tracking as htm

# Configuration constants
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
HISTORY_LENGTH = 3             # History length for gesture smoothing
COOLDOWN_MS = 500              # Cooldown time between gestures
CONFIDENCE_THRESHOLD = 0.65    # Confidence threshold for gesture recognition
SWIPE_PIXELS = 100             # Minimum pixels for swipe gesture
SWIPE_WINDOW_MS = 220          # Time window for swipe gesture
FIST_TO_OPEN_MS = 900          # Transition time from fist to open hand

# Finger landmark IDs
FINGER_TIP_IDS = [4, 8, 12, 16, 20]    # Thumb, Index, Middle, Ring, Pinky
FINGER_PIP_IDS = [3, 6, 10, 14, 18]


def fingers_up(lm_list, hand_label: str):
    pts = {i:(x, y) for (i, x, y) in lm_list}
    up = [0, 0, 0, 0, 0]

    if 4 in pts and 3 in pts:
        if hand_label == "Right":
            up[0] = 1 if pts[4][0] < pts[3][0] else 0
        else:
            up[0] = 1 if pts[4][0] > pts[3][0] else 0

    for finger_index, (tip, pip) in enumerate(zip(FINGER_TIP_IDS[1:], FINGER_PIP_IDS[1:]), start=1):
        if tip in pts and pip in pts:
            up[finger_index] = 1 if pts[tip][1] < pts[pip][1] else 0

    distances = []
    for tip, pip in zip(FINGER_TIP_IDS[1:], FINGER_PIP_IDS[1:]):
        if tip in pts and pip in pts:
            distances.append(abs(pts[pip][1] - pts[tip][1]))
    confidence = min(1.0, (np.mean(distances) / 40.0) if distances else 0.0)
    return up, confidence


def classify_gesture(landmark_list, hand_label: str):
    x_coordinates = [x for (_, x, _) in landmark_list]
    center_x = int(sum(x_coordinates) / len(x_coordinates)) if x_coordinates else None

    fingers_up_state, confidence = fingers_up(landmark_list, hand_label)

    if fingers_up_state[1] == 1 and fingers_up_state[2] == 1 and fingers_up_state[3] == 0 and fingers_up_state[4] == 0:
        return "two_fingers", max(0.5, confidence), center_x

    if sum(fingers_up_state) >= 4:
        return "open_palm", max(0.5, confidence), center_x

    if sum(fingers_up_state[1:]) == 0:
        return "fist", max(0.5, 1.0 - confidence / 1.2), center_x

    return None, 0.0, center_x


def emit(gesture, confidence, hand="Right"):
    now = int(time.time() * 1000)
    event = {"gesture": gesture, "confidence": round(float(confidence), 3), "hand": hand, "ts": int(now/1000)}
    print(json.dumps(event), flush=True)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    detector = htm.HandDetector(detection_con=0.6, track_con=0.5, max_hands=1)

    history = collections.deque(maxlen=HISTORY_LENGTH)
    recent_positions = collections.deque()
    last_state = None
    last_state_time = 0
    last_emit_time = 0

    previous_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        detector.find_hands(img, draw=True)
        hands = detector.get_hands(img, draw_dots=False)

        current_time_ms = int(time.time() * 1000)

        gesture_label, gesture_confidence, center_x, hand_label = None, 0.0, None, "Right"
        if hands:
            hand = hands[0]
            hand_label = hand.label
            gesture_label, gesture_confidence, center_x = classify_gesture(hand.lm_list, hand_label)
            debug_fingers_up, _ = fingers_up(hand.lm_list, hand_label)
            cv2.putText(img, f"UP={debug_fingers_up} {gesture_label}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1)

        history.append((gesture_label, gesture_confidence, current_time_ms))

        if center_x is not None:
            recent_positions.append((current_time_ms, center_x))

        while recent_positions and current_time_ms - recent_positions[0][0] > SWIPE_WINDOW_MS:
            recent_positions.popleft()

        is_stable = False
        average_confidence = 0.0
        if len(history) == history.maxlen:
            labels = [label for (label, _, _) in history]
            confidences = [conf for (label, conf, _) in history if label]
            is_stable = (labels.count(labels[-1]) == len(labels) and labels[-1] is not None)
            average_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0

        if gesture_label == "fist":
            last_state, last_state_time = "fist", current_time_ms

        if gesture_label == "open_palm" and last_state == "fist" and (current_time_ms - last_state_time) <= FIST_TO_OPEN_MS:
            if gesture_confidence >= (CONFIDENCE_THRESHOLD - 0.05) and (current_time_ms - last_emit_time) >= COOLDOWN_MS:
                emit("play_pause", gesture_confidence, hand_label)
                last_emit_time = current_time_ms
            last_state = None

        if is_stable and history[-1][0] == "two_fingers" and len(recent_positions) >= 2:
            delta_x = recent_positions[-1][1] - recent_positions[0][1]
            if abs(delta_x) >= SWIPE_PIXELS and average_confidence >= (CONFIDENCE_THRESHOLD - 0.1):
                if current_time_ms - last_emit_time >= COOLDOWN_MS:
                    emit("prev" if delta_x > 0 else "next", average_confidence, hand_label)
                    last_emit_time = current_time_ms
                    recent_positions.clear()

        current_time = time.time()
        fps = 1 / max(1e-6, (current_time - previous_time))
        previous_time = current_time
        hud_text = f"{int(fps)} FPS | label={gesture_label or '-'} conf={average_confidence:.2f}"
        cv2.putText(img, hud_text, (10, 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

        cv2.imshow("Recognizer", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()