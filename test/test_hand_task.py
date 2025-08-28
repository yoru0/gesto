import cv2
import time
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

WINDOW_NAME = "Hand Landmarker"
MODEL_PATH = "test\hand_landmarker.task"
CAM_INDEX = 0

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=VisionRunningMode.VIDEO,
)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

t0 = time.time()
frames = 0
fps = 0.0

def swap_handedness(raw: str) -> str:
    if raw == "Left":
        return "Right"
    if raw == "Right":
        return "Left"
    return raw or ""

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

            if result and result.hand_landmarks:
                for i, lm_list in enumerate(result.hand_landmarks):
                    proto = landmark_pb2.NormalizedLandmarkList()
                    proto.landmark.extend(
                        landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in lm_list
                    )

                    drawing_utils.draw_landmarks(
                        frame_bgr, proto, HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

                    raw = ""
                    try:
                        raw = result.handedness[i][0].category_name
                    except Exception:
                        pass
                    handed = swap_handedness(raw)

                    h, w, _ = frame_bgr.shape
                    xs = [l.x for l in lm_list]
                    ys = [l.y for l in lm_list]
                    x_px = int(min(xs) * w)
                    y_px = int(min(ys) * h) - 10
                    cv2.putText(
                        frame_bgr, handed, (x_px, max(18, y_px)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                    )

            frames += 1
            if frames >= 10:
                t1 = time.time()
                fps = frames / (t1 - t0)
                t0, frames = t1, 0
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame_bgr)

            # 1) Pump events + hotkeys
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break

            # 2) Detect window close (❌). If not visible, break.
            #    On many builds this returns < 1 when the user clicks X.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

finally:
    cap.release()
    cv2.destroyAllWindows()