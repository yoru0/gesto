import cv2
import numpy as np
import json, time, collections
import hand_tracking_module as htm

# Tunables
CAM_W, CAM_H    = 640, 480
HIST            = 3             # History length for gesture smoothing
COOLDOWN_MS     = 500           # Cooldown time between gestures
CONF_THRESH     = 0.65          # Confidence threshold for gesture recognition
SWIPE_PIXELS    = 100           # Minimum pixels for swipe gesture
SWIPE_WINDOW_MS = 220           # Time window for swipe gesture
FIST_TO_OPEN_MS = 900           # Transition time from fist to open hand

# Finger IDs
TIP_IDS = [4, 8, 12, 16, 20]    # Thumb, Index, Middle, Ring, Pinky
PIP_IDS = [3, 6, 10, 14, 18]


def fingers_up(lm_list, hand_label: str):
    pts = {i:(x, y) for (i, x, y) in lm_list}
    up = [0, 0, 0, 0, 0]

    if 4 in pts and 3 in pts:
        if hand_label == "Right":
            up[0] = 1 if pts[4][0] < pts[3][0] else 0
        else:
            up[0] = 1 if pts[4][0] > pts[3][0] else 0

    for fi, (tip, pip) in enumerate(zip(TIP_IDS[1:], PIP_IDS[1:]), start=1):
        if tip in pts and pip in pts:
            up[fi] = 1 if pts[tip][1] < pts[pip][1] else 0

    dists = []
    for tip, pip in zip(TIP_IDS[1:], PIP_IDS[1:]):
        if tip in pts and pip in pts:
            dists.append(abs(pts[pip][1] - pts[tip][1]))
    conf = min(1.0, (np.mean(dists) / 40.0) if dists else 0.0)
    return up, conf


def classify_gesture(lm_list, hand_label: str):
    xs = [x for (_, x, _) in lm_list]
    cx = int(sum(xs) / len(xs)) if xs else None

    up, conf = fingers_up(lm_list, hand_label)

    if up[1] == 1 and up[2] == 1 and up[3] == 0 and up[4] == 0:
        return "two_fingers", max(0.5, conf), cx

    if sum(up) >= 4:
        return "open_palm", max(0.5, conf), cx

    if sum(up[1:]) == 0:
        return "fist", max(0.5, 1.0 - conf / 1.2), cx

    return None, 0.0, cx


def emit(gesture, conf, hand="Right"):
    now = int(time.time() * 1000)
    evt = {"gesture": gesture, "confidence": round(float(conf), 3), "hand": hand, "ts": int(now/1000)}
    print(json.dumps(evt), flush=True)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    detector = htm.HandDetector(detection_con=0.6, track_con=0.5, max_hands=1)

    history = collections.deque(maxlen=HIST)
    recent_positions = collections.deque()
    last_state = None
    last_state_time = 0
    last_emit_time = 0

    prev_time = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        detector.find_hands(img, draw=True)
        hands = detector.get_hands(img, draw_dots=False)

        t_ms = int(time.time() * 1000)

        label, conf, cx, hand_lbl = None, 0.0, None, "Right"
        if hands:
            h = hands[0]
            hand_lbl = h.label
            label, conf, cx = classify_gesture(h.lm_list, hand_lbl)
            dbg_up, _ = fingers_up(h.lm_list, hand_lbl)
            cv2.putText(img, f"UP={dbg_up} {label}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1)

        history.append((label, conf, t_ms))

        if cx is not None:
            recent_positions.append((t_ms, cx))

        while recent_positions and t_ms - recent_positions[0][0] > SWIPE_WINDOW_MS:
            recent_positions.popleft()

        stable = False
        avg_conf = 0.0
        if len(history) == history.maxlen:
            labels = [l for (l, _, _) in history]
            confs  = [c for (l, c, _) in history if l]
            stable = (labels.count(labels[-1]) == len(labels) and labels[-1] is not None)
            avg_conf = (sum(confs) / len(confs)) if confs else 0.0

        if label == "fist":
            last_state, last_state_time = "fist", t_ms

        if label == "open_palm" and last_state == "fist" and (t_ms - last_state_time) <= FIST_TO_OPEN_MS:
            if conf >= (CONF_THRESH - 0.05) and (t_ms - last_emit_time) >= COOLDOWN_MS:
                emit("play_pause", conf, hand_lbl)
                last_emit_time = t_ms
            last_state = None

        if stable and history[-1][0] == "two_fingers" and len(recent_positions) >= 2:
            dx = recent_positions[-1][1] - recent_positions[0][1]
            if abs(dx) >= SWIPE_PIXELS and avg_conf >= (CONF_THRESH - 0.1):
                if t_ms - last_emit_time >= COOLDOWN_MS:
                    emit("prev" if dx > 0 else "next", avg_conf, hand_lbl)
                    last_emit_time = t_ms
                    recent_positions.clear()

        curr_time = time.time()
        fps = 1 / max(1e-6, (curr_time - prev_time))
        prev_time = curr_time
        hud = f"{int(fps)} FPS | label={label or '-'} conf={avg_conf:.2f}"
        cv2.putText(img, hud, (10, 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

        cv2.imshow("Recognizer", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()